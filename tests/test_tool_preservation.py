#!/usr/bin/env python3
"""
Test tool calls preservation with guardrails - Unit tests with mocking
"""
import pytest
from unittest.mock import AsyncMock, patch
from haliosai import guarded_chat_completion


class TestToolPreservation:
    """Test that tool calls are preserved through guardrail evaluation"""

    @pytest.fixture
    def mock_guardrail_response_success(self):
        """Mock successful guardrail response"""
        return {
            "guardrails_triggered": 0,
            "violations": [],
            "evaluation_time": 0.05
        }

    @pytest.fixture
    def mock_guardrail_response_violation(self):
        """Mock guardrail response with violations"""
        return {
            "guardrails_triggered": 1,
            "result": [{"triggered": True, "guardrail_type": "content_policy", "severity": "high"}],
            "evaluation_time": 0.05
        }

    def test_mock_response_structure(self):
        """Test that mock response has correct structure"""
        # Mock LLM response that includes tool calls
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()

        class MockMessage:
            def __init__(self):
                self.content = None
                self.tool_calls = [MockToolCall()]

        class MockToolCall:
            def __init__(self):
                self.id = "call_123"
                self.type = "function"
                self.function = MockFunction()

        class MockFunction:
            def __init__(self):
                self.name = "calculate_math"
                self.arguments = '{"expression":"15 * 8 + 42"}'

        response = MockResponse()

        # Verify structure
        assert hasattr(response, 'choices')
        assert len(response.choices) == 1

        message = response.choices[0].message
        assert message.content is None
        assert hasattr(message, 'tool_calls')
        assert len(message.tool_calls) == 1

        tool_call = message.tool_calls[0]
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "calculate_math"
        assert tool_call.function.arguments == '{"expression":"15 * 8 + 42"}'

    @pytest.mark.asyncio
    async def test_decorator_with_mock_llm_success(self, mock_guardrail_response_success):
        """Test decorator with successful guardrail evaluation"""
        from haliosai.client import HaliosGuard

        # Create a mock LLM function
        async def mock_llm(_messages):
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]

            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()

            class MockMessage:
                def __init__(self):
                    self.content = None
                    self.tool_calls = [MockToolCall()]

            class MockToolCall:
                def __init__(self):
                    self.id = "call_123"
                    self.type = "function"
                    self.function = MockFunction()

            class MockFunction:
                def __init__(self):
                    self.name = "calculate_math"
                    self.arguments = '{"expression":"15 * 8 + 42"}'

            return MockResponse()

        messages = [{"role": "user", "content": "Calculate 15 * 8 + 42"}]

        # Test the HaliosGuard directly instead of the decorator
        config = {
            'agent_id': "test-agent",
            'api_key': "test-key",
            'base_url': "http://test-url",
            'parallel': False,
            'streaming': False
        }

        guard = HaliosGuard(**config)
        with patch.object(guard, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluate.return_value = mock_guardrail_response_success

            async with guard:
                result = await guard.guarded_call_parallel(messages, mock_llm)

            # Verify evaluate was called (should be called twice: request and response)
            assert mock_evaluate.call_count == 2

            # Verify result structure
            assert result.result.name == "SUCCESS"
            assert result.final_response is not None
            llm_response = result.final_response
            assert hasattr(llm_response, 'choices')
            assert llm_response.choices is not None
            assert len(llm_response.choices) == 1

            message = llm_response.choices[0].message

            # Verify tool calls are preserved
            assert message.tool_calls is not None
            assert len(message.tool_calls) == 1
            tool_call = message.tool_calls[0]
            assert tool_call.id == "call_123"
            assert tool_call.type == "function"
            assert tool_call.function.name == "calculate_math"
            assert tool_call.function.arguments == '{"expression":"15 * 8 + 42"}'

    @pytest.mark.asyncio
    async def test_decorator_with_request_violation(self, mock_guardrail_response_violation):
        """Test when request guardrails are triggered"""
        from haliosai.client import HaliosGuard

        async def mock_llm(_messages):
            return {"error": "Should not reach here"}

        messages = [{"role": "user", "content": "Inappropriate content"}]

        config = {
            'agent_id': "test-agent",
            'api_key': "test-key",
            'base_url': "http://test-url",
            'parallel': False,
            'streaming': False
        }

        guard = HaliosGuard(**config)
        with patch.object(guard, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluate.return_value = mock_guardrail_response_violation

            # Should return a blocked response due to guardrail violation
            async with guard:
                result = await guard.guarded_call_parallel(messages, mock_llm)

            assert result.result.name == "REQUEST_BLOCKED"
            assert len(result.request_violations) == 1
            assert result.request_violations[0].guardrail_type == "content_policy"

    @pytest.mark.asyncio
    async def test_decorator_with_response_violation(self, mock_guardrail_response_success, mock_guardrail_response_violation):
        """Test when response guardrails are triggered"""
        from haliosai.client import HaliosGuard

        async def mock_llm(_messages):
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]

            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()

            class MockMessage:
                def __init__(self):
                    self.content = "This response violates policy"
                    self.tool_calls = []

            return MockResponse()

        messages = [{"role": "user", "content": "Generate inappropriate content"}]

        config = {
            'agent_id': "test-agent",
            'api_key': "test-key",
            'base_url': "http://test-url",
            'parallel': False,
            'streaming': False
        }

        guard = HaliosGuard(**config)
        with patch.object(guard, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            # First call (request) succeeds, second call (response) fails
            mock_evaluate.side_effect = [
                mock_guardrail_response_success,  # Request check
                mock_guardrail_response_violation  # Response check
            ]

            # Should return a blocked response due to response guardrail violation
            async with guard:
                result = await guard.guarded_call_parallel(messages, mock_llm)

            assert result.result.name == "RESPONSE_BLOCKED"
            assert len(result.response_violations) == 1
            assert result.response_violations[0].guardrail_type == "content_policy"

    def test_decorator_initialization(self):
        """Test decorator initialization with various parameters"""

        # Test with minimal parameters
        decorator = guarded_chat_completion(agent_id="test-agent")
        assert decorator is not None

        # Test with all parameters
        decorator_full = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test-url"
        )
        assert decorator_full is not None

    @pytest.mark.asyncio
    async def test_tool_calls_preservation_through_guardrails(self, mock_guardrail_response_success):
        """Test that tool calls are preserved when guardrails pass"""
        from haliosai.client import HaliosGuard

        async def mock_llm_with_tools(_messages):
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]

            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()

            class MockMessage:
                def __init__(self):
                    self.content = None
                    self.tool_calls = [
                        MockToolCall("call_1", "get_weather", '{"location": "NYC"}'),
                        MockToolCall("call_2", "get_time", '{"timezone": "EST"}')
                    ]

            class MockToolCall:
                def __init__(self, call_id, func_name, args):
                    self.id = call_id
                    self.type = "function"
                    self.function = MockFunction(func_name, args)

            class MockFunction:
                def __init__(self, name, args):
                    self.name = name
                    self.arguments = args

            return MockResponse()

        messages = [{"role": "user", "content": "Get weather and time for NYC"}]

        config = {
            'agent_id': "test-agent",
            'api_key': "test-key",
            'base_url': "http://test-url",
            'parallel': False,
            'streaming': False
        }

        guard = HaliosGuard(**config)
        with patch.object(guard, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluate.return_value = mock_guardrail_response_success

            async with guard:
                result = await guard.guarded_call_parallel(messages, mock_llm_with_tools)

            # Verify result structure
            assert result.result.name == "SUCCESS"
            assert result.final_response is not None
            llm_response = result.final_response
            assert hasattr(llm_response, 'choices')
            assert llm_response.choices is not None
            assert len(llm_response.choices) == 1

            message = llm_response.choices[0].message
            assert len(message.tool_calls) == 2
            assert message.tool_calls[0].function.name == "get_weather"
            assert message.tool_calls[1].function.name == "get_time"
