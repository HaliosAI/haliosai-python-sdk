"""
OpenAI Agents Framework Integration for HaliosAI

This module provides native guardrail implementations for the OpenAI Agents framework.
Instead of patching the OpenAI client, users can simply add these guardrails to their Agent definitions.

Usage:
    from haliosai.openai import HaliosInputGuardrail, HaliosOutputGuardrail
    from agents import Agent
    
    agent = Agent(
        name="my_agent",
        instructions="You are a helpful assistant",
        input_guardrails=[HaliosInputGuardrail(agent_id="your-agent-id")],
        output_guardrails=[HaliosOutputGuardrail(agent_id="your-agent-id")]
    )
"""

from __future__ import annotations

import logging
from typing import Any, Union

from .client import HaliosGuard
from .config import get_api_key, get_base_url

try:
    from agents.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
    from agents.run_context import RunContextWrapper
    from agents.items import TResponseInputItem
    from agents.agent import Agent
    AGENTS_AVAILABLE = True
except ImportError:
    # Define mock classes for when agents framework is not installed
    class GuardrailFunctionOutput:
        def __init__(self, output_info: Any, tripwire_triggered: bool):
            self.output_info = output_info
            self.tripwire_triggered = tripwire_triggered
    
    class InputGuardrail:
        def __init__(self, guardrail_function, name=None):
            self.guardrail_function = guardrail_function
            self.name = name
    
    class OutputGuardrail:
        def __init__(self, guardrail_function, name=None):
            self.guardrail_function = guardrail_function
            self.name = name
    
    class RunContextWrapper:
        pass
    
    TResponseInputItem = Any
    Agent = Any
    AGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HaliosInputGuardrail(InputGuardrail):
    """
    HaliosAI Input Guardrail for OpenAI Agents Framework
    
    Evaluates user input through the Halios backend before the agent processes it.
    If the guardrail is triggered, the agent execution will be halted.
    
    Args:
        agent_id: Your Halios agent ID
        api_key: Halios API key (optional, uses environment variable if not provided)
        base_url: Halios API base URL (optional, defaults to https://api.halios.ai)
        name: Guardrail name for tracing (optional)
    
    Example:
        ```python
        from haliosai.openai import HaliosInputGuardrail
        from agents import Agent
        
        agent = Agent(
            name="my_agent",
            instructions="You are a helpful assistant",
            input_guardrails=[HaliosInputGuardrail(agent_id="your-agent-id")]
        )
        ```
    """
    
    def __init__(
        self,
        agent_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None
    ):
        if not AGENTS_AVAILABLE:
            raise ImportError(
                "OpenAI Agents framework is not installed. "
                "Install it with: pip install openai-agents"
            )
        
        self.agent_id = agent_id
        self.api_key = api_key
        self.base_url = base_url
        self._name = name
        
        # Initialize HaliosGuard client
        self._guard = HaliosGuard(
            agent_id=self.agent_id,
            api_key=self.api_key or get_api_key() or "",
            base_url=self.base_url or get_base_url()
        )
        
        # Set up guardrail function for the parent class
        super().__init__(
            guardrail_function=self._evaluate_input,
            name=self.get_name()
        )
    
    def get_name(self) -> str:
        return self._name or f"halios_input_{self.agent_id}"
    
    async def _evaluate_input(
        self,
        context: RunContextWrapper,
        agent: Agent,
        input_data: Union[str, list[TResponseInputItem]]
    ) -> GuardrailFunctionOutput:
        """
        Evaluate input through Halios guardrails
        """
        try:
            # Convert input to string if needed
            if isinstance(input_data, list):
                # Extract text from response input items
                input_text = ""
                for item in input_data:
                    if hasattr(item, 'content') and isinstance(item.content, str):
                        input_text += item.content + "\n"
                    elif hasattr(item, 'text') and isinstance(item.text, str):
                        input_text += item.text + "\n"
                    elif isinstance(item, dict) and 'content' in item:
                        input_text += str(item['content']) + "\n"
                    else:
                        input_text += str(item) + "\n"
                input_text = input_text.strip()
            else:
                input_text = str(input_data)
            
            logger.debug("🔍 HaliosAI Input Guardrail: Evaluating input for agent %s", self.agent_id)
            
            # Directly call the backend API to evaluate guardrails
            messages = [{"role": "user", "content": input_text}]
            backend_result = await self._guard.evaluate(messages, invocation_type="request")
            
            # Simple check: are any guardrails triggered?
            triggered = backend_result.get("guardrails_triggered", 0) > 0
            
            if triggered:
                logger.warning("🚫 HaliosAI Input Guardrail: Tripwire triggered for agent %s", self.agent_id)
                output_info = {
                    "guardrail_type": "halios_input",
                    "agent_id": self.agent_id,
                    "triggered": True,
                    "details": f"Input blocked: {backend_result.get('guardrails_triggered', 0)} guardrails triggered",
                    "backend_response": backend_result
                }
            else:
                logger.debug("✅ HaliosAI Input Guardrail: Input approved for agent %s", self.agent_id)
                output_info = {
                    "guardrail_type": "halios_input",
                    "agent_id": self.agent_id,
                    "triggered": False,
                    "details": "Input approved"
                }
            
            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=triggered  # Simple and direct!
            )
            
        except Exception as e:
            logger.error("❌ HaliosAI Input Guardrail Error for agent %s: %s", self.agent_id, e)
            # On error, allow the request to proceed but log the issue
            return GuardrailFunctionOutput(
                output_info={
                    "guardrail_type": "halios_input",
                    "agent_id": self.agent_id,
                    "triggered": False,
                    "error": str(e),
                    "details": "Guardrail evaluation failed, allowing request"
                },
                tripwire_triggered=False
            )


class HaliosOutputGuardrail(OutputGuardrail):
    """
    HaliosAI Output Guardrail for OpenAI Agents Framework
    
    Evaluates agent output through the Halios backend after the agent generates a response.
    If the guardrail is triggered, an exception will be raised.
    
    Args:
        agent_id: Your Halios agent ID
        api_key: Halios API key (optional, uses environment variable if not provided)
        base_url: Halios API base URL (optional, defaults to https://api.halios.ai)
        name: Guardrail name for tracing (optional)
    
    Example:
        ```python
        from haliosai.openai import HaliosOutputGuardrail
        from agents import Agent
        
        agent = Agent(
            name="my_agent",
            instructions="You are a helpful assistant",
            output_guardrails=[HaliosOutputGuardrail(agent_id="your-agent-id")]
        )
        ```
    """
    
    def __init__(
        self,
        agent_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None
    ):
        if not AGENTS_AVAILABLE:
            raise ImportError(
                "OpenAI Agents framework is not installed. "
                "Install it with: pip install openai-agents"
            )
        
        self.agent_id = agent_id
        self.api_key = api_key
        self.base_url = base_url
        self._name = name
        
        # Initialize HaliosGuard client
        self._guard = HaliosGuard(
            agent_id=self.agent_id,
            api_key=self.api_key or get_api_key() or "",
            base_url=self.base_url or get_base_url()
        )
        
        # Set up guardrail function for the parent class
        super().__init__(
            guardrail_function=self._evaluate_output,
            name=self.get_name()
        )
    
    def get_name(self) -> str:
        return self._name or f"halios_output_{self.agent_id}"
    
    async def _evaluate_output(
        self,
        context: RunContextWrapper,
        agent: Agent,
        agent_output: Any
    ) -> GuardrailFunctionOutput:
        """
        Evaluate output through Halios guardrails
        """
        try:
            # Convert output to string
            if isinstance(agent_output, str):
                output_text = agent_output
            elif hasattr(agent_output, 'content'):
                output_text = str(agent_output.content)
            elif hasattr(agent_output, 'text'):
                output_text = str(agent_output.text)
            elif isinstance(agent_output, dict):
                output_text = str(agent_output.get('content', str(agent_output)))
            else:
                output_text = str(agent_output)
            
            logger.debug("🔍 HaliosAI Output Guardrail: Evaluating output for agent %s", self.agent_id)
            
            # For output evaluation, we need both input and output
            # Try to get the input from the context if available
            input_text = ""
            # Simple approach: just use empty input for now since we have the output
            
            # Directly call the backend API to evaluate guardrails
            messages = [
                {"role": "user", "content": input_text or "Previous conversation"},
                {"role": "assistant", "content": output_text}
            ]
            backend_result = await self._guard.evaluate(messages, invocation_type="response")
            
            # Simple check: are any guardrails triggered?
            triggered = backend_result.get("guardrails_triggered", 0) > 0

            if triggered:
                logger.warning("🚫 HaliosAI Output Guardrail: Tripwire triggered for agent %s", self.agent_id)
                output_info = {
                    "guardrail_type": "halios_output",
                    "agent_id": self.agent_id,
                    "triggered": True,
                    "details": f"Output blocked: {backend_result.get('guardrails_triggered', 0)} guardrails triggered",
                    "backend_response": backend_result
                }
            else:
                logger.debug("✅ HaliosAI Output Guardrail: Output approved for agent %s", self.agent_id)
                output_info = {
                    "guardrail_type": "halios_output",
                    "agent_id": self.agent_id,
                    "triggered": False,
                    "details": "Output approved"
                }
            
            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=triggered  # Simple and direct!
            )
            
        except Exception as e:
            logger.error("❌ HaliosAI Output Guardrail Error for agent %s: %s", self.agent_id, e)
            # On error, allow the response but log the issue
            return GuardrailFunctionOutput(
                output_info={
                    "guardrail_type": "halios_output",
                    "agent_id": self.agent_id,
                    "triggered": False,
                    "error": str(e),
                    "details": "Guardrail evaluation failed, allowing response"
                },
                tripwire_triggered=False
            )


# Convenience aliases for easier imports
RemoteInputGuardrail = HaliosInputGuardrail
RemoteOutputGuardrail = HaliosOutputGuardrail
