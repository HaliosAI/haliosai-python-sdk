"""mount_halios() — attach HaliosOptimizable routes to a FastAPI app.

Usage::

    from fastapi import FastAPI
    from haliosai.optimize.server import mount_halios

    app = FastAPI()

    # Your agent state
    _current_prompt = "You are a helpful assistant."
    _conversations: dict[str, list[dict]] = {}

    async def agent_fn(messages: list[dict], prompt: str) -> str:
        \"\"\"Call your LLM/chatbot and return the assistant response text.\"\"\"
        ...

    def prompt_getter() -> str:
        return _current_prompt

    def prompt_setter(new_prompt: str) -> None:
        global _current_prompt
        _current_prompt = new_prompt

    mount_halios(app, agent_fn=agent_fn, prompt_getter=prompt_getter, prompt_setter=prompt_setter)
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


class _ChatRequest(BaseModel):
    conversation_id: str
    messages: list[dict[str, str]]


class _ChatResponse(BaseModel):
    conversation_id: str
    response: str
    trace_messages: list[dict[str, Any]] = Field(default_factory=list)


class _ConfigResponse(BaseModel):
    current_prompt: str
    supported_scenarios: list[str] = []
    tool_schemas: list[dict[str, Any]] = []


def mount_halios(
    app: Any,
    *,
    agent_fn: Callable[..., Any],
    prompt_getter: Callable[[], str],
    prompt_setter: Callable[[str], None],
    supported_scenarios: list[str] | None = None,
) -> None:
    """Mount /halios/* routes onto a FastAPI (or Starlette) application.

    Args:
        app: The FastAPI app instance to mount routes onto.
        agent_fn: ``async (messages: list[dict], prompt: str) -> str``
            Called with the full conversation history and the current prompt.
            Should return the assistant text for the last turn.
        prompt_getter: ``() -> str``
            Returns the current system prompt.
        prompt_setter: ``(new_prompt: str) -> None``
            Persists an updated system prompt (called by the optimizer).
        supported_scenarios: Optional list of scenario IDs this agent supports.
            Returned in GET /halios/config for the optimizer to query.
    """
    try:
        from fastapi import APIRouter, Header
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required to use mount_halios(). "
            "Install it with: pip install fastapi"
        ) from exc

    _conversations: dict[str, list[dict[str, str]]] = {}
    _router = APIRouter(prefix="/halios", tags=["halios-optimize"])

    @_router.get("/config", response_model=_ConfigResponse)
    async def get_config() -> _ConfigResponse:
        return _ConfigResponse(
            current_prompt=prompt_getter(),
            supported_scenarios=supported_scenarios or [],
        )

    @_router.post("/config")
    async def set_config(body: dict[str, Any]) -> dict[str, str]:
        new_prompt: str | None = body.get("current_prompt")
        if new_prompt is not None:
            prompt_setter(new_prompt)
        return {"status": "ok"}

    @_router.post("/chat", response_model=_ChatResponse)
    async def chat(
        request: _ChatRequest,
        x_halios_prompt_b64: str | None = Header(default=None),
        x_halios_prompt: str | None = Header(default=None),
    ) -> _ChatResponse:
        conv_id = request.conversation_id
        if x_halios_prompt_b64 is not None:
            import base64
            prompt = base64.b64decode(x_halios_prompt_b64.encode()).decode()
        elif x_halios_prompt is not None:
            prompt = x_halios_prompt
        else:
            prompt = prompt_getter()

        history = _conversations.get(conv_id, [])
        new_messages = [m for m in request.messages if m not in history]
        history = history + new_messages
        _conversations[conv_id] = history

        import asyncio
        import inspect

        if inspect.iscoroutinefunction(agent_fn):
            agent_result = await agent_fn(history, prompt)
        else:
            loop = asyncio.get_event_loop()
            agent_result = await loop.run_in_executor(None, agent_fn, history, prompt)

        trace_messages: list[dict[str, Any]] = []
        if isinstance(agent_result, dict):
            response_text = str(agent_result.get("response") or "")
            raw_trace_messages = agent_result.get("trace_messages")
            if isinstance(raw_trace_messages, list):
                trace_messages = [message for message in raw_trace_messages if isinstance(message, dict)]
        else:
            response_text = str(agent_result or "")

        if trace_messages:
            has_visible_assistant = any(
                message.get("role") == "assistant" and not message.get("tool_calls")
                for message in trace_messages
            )
            if response_text and not has_visible_assistant:
                trace_messages.append({"role": "assistant", "content": response_text})
            _conversations[conv_id] = history + trace_messages
        else:
            _conversations[conv_id] = history + [{"role": "assistant", "content": response_text}]

        return _ChatResponse(
            conversation_id=conv_id,
            response=response_text,
            trace_messages=trace_messages,
        )

    @_router.delete("/conversation/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, str]:
        _conversations.pop(conversation_id, None)
        return {"status": "deleted"}

    app.include_router(_router)
