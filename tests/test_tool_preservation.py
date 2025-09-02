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
            "violations": [{"type": "content_policy", "severity": "high"}],
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

        # Apply decorator with test configuration
        decorated_llm = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test-url",
            streaming_guardrails=False
        )(mock_llm)

        messages = [{"role": "user", "content": "Calculate 15 * 8 + 42"}]

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            # Create a proper mock response
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_guardrail_response_success)
            mock_response.raise_for_status = AsyncMock()
            mock_post.return_value = mock_response

            result = await decorated_llm(messages)

            # Verify API was called (should be called twice: request and response)
            assert mock_post.call_count == 2

            # Verify result structure - GuardedResponse contains final_response
            assert hasattr(result, 'final_response')
            assert result.final_response is not None
            
            # The final_response should be the original LLM response with tool calls
            llm_response = result.final_response
            assert hasattr(llm_response, 'choices')
            assert len(llm_response.choices) == 1

            message = llm_response.choices[0].message
            assert message.content is None
            assert hasattr(message, 'tool_calls')
            assert len(message.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_decorator_with_request_violation(self, mock_guardrail_response_violation):
        """Test decorator when request guardrails are triggered"""

        async def mock_llm(_messages):
            return {"error": "Should not reach here"}

        decorated_llm = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key"
        )(mock_llm)

        messages = [{"role": "user", "content": "Inappropriate content"}]

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = AsyncMock()
            mock_post.return_value.json = AsyncMock(return_value=mock_guardrail_response_violation)

            # Should raise an exception due to guardrail violation
            with pytest.raises(ValueError):
                await decorated_llm(messages)

    @pytest.mark.asyncio
    async def test_decorator_with_response_violation(self, mock_guardrail_response_success, mock_guardrail_response_violation):
        """Test decorator when response guardrails are triggered"""

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

        decorated_llm = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key"
        )(mock_llm)

        messages = [{"role": "user", "content": "Generate inappropriate content"}]

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            # First call (request) succeeds, second call (response) fails
            mock_post.return_value = AsyncMock()
            mock_post.return_value.json = AsyncMock(side_effect=[
                mock_guardrail_response_success,  # Request check
                mock_guardrail_response_violation  # Response check
            ])

            # Should raise an exception due to response guardrail violation
            with pytest.raises(ValueError):
                await decorated_llm(messages)

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

        decorated_llm = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key"
        )(mock_llm_with_tools)

        messages = [{"role": "user", "content": "Get weather and time for NYC"}]

        with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = AsyncMock()
            mock_post.return_value.json = AsyncMock(return_value=mock_guardrail_response_success)

            result = await decorated_llm(messages)

            # Verify tool calls are preserved
            assert hasattr(result, 'final_response')
            llm_response = result.final_response
            assert hasattr(llm_response, 'choices')
            
            message = llm_response.choices[0].message
            assert len(message.tool_calls) == 2
            assert message.tool_calls[0].function.name == "get_weather"
            assert message.tool_calls[1].function.name == "get_time"
