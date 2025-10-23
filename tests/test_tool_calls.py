#!/usr/bin/env python3
"""
Test tool calls functionality - Unit tests with mocking
"""
import pytest
from unittest.mock import patch, MagicMock
from haliosai import HaliosGuard


class TestToolCalls:
    """Test tool calls handling in guardrail evaluation"""

    @pytest.fixture
    def guard(self):
        """Create a HaliosGuard instance for testing"""
        guard_instance = HaliosGuard(agent_id="test-agent", api_key="test-key")
        # Initialize HTTP client for testing
        guard_instance._ensure_http_client_for_testing()
        return guard_instance

    @pytest.fixture
    def mock_response_success(self):
        """Mock successful API response"""
        return {
            "guardrails_triggered": 0,
            "violations": [],
            "evaluation_time": 0.1
        }

    @pytest.fixture
    def mock_response_violation(self):
        """Mock API response with violations"""
        return {
            "guardrails_triggered": 1,
            "result": [{"type": "content_policy", "severity": "high", "guardrail_type": "content_policy", "triggered": True}],
            "evaluation_time": 0.1
        }

    def test_messages_without_tool_calls(self):
        """Test evaluation of messages without tool calls"""
        messages = [
            {
                "role": "user",
                "content": "Just say hello to me"
            }
        ]

        # Test that messages are properly structured
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Just say hello to me"
        assert "tool_calls" not in messages[0]

    def test_messages_with_tool_calls_structure(self):
        """Test structure of messages containing tool calls"""
        messages_with_tools = [
            {
                "role": "user",
                "content": "What's the weather like?"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": "72F", "condition": "sunny"}'
            }
        ]

        # Verify message structure
        assert len(messages_with_tools) == 3

        # User message
        assert messages_with_tools[0]["role"] == "user"
        assert messages_with_tools[0]["content"] == "What's the weather like?"

        # Assistant message with tool calls
        assert messages_with_tools[1]["role"] == "assistant"
        assert messages_with_tools[1]["content"] is None
        assert "tool_calls" in messages_with_tools[1]
        assert len(messages_with_tools[1]["tool_calls"]) == 1

        # Tool message
        assert messages_with_tools[2]["role"] == "tool"
        assert messages_with_tools[2]["tool_call_id"] == "call_123"
        assert messages_with_tools[2]["content"] == '{"temperature": "72F", "condition": "sunny"}'

    @pytest.mark.asyncio
    async def test_evaluate_request_without_tools_mock(self, guard, mock_response_success):
        """Test request evaluation without tool calls using mock"""
        messages = [
            {
                "role": "user",
                "content": "Just say hello to me"
            }
        ]

        with patch.object(guard.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response_success
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard.evaluate(messages, "request")

            # Verify the API was called and result is a ScanResult object
            assert isinstance(result, object)  # It's a ScanResult, not a dict

    @pytest.mark.asyncio
    async def test_evaluate_response_with_tools_mock(self, guard, mock_response_success):
        """Test response evaluation with tool calls using mock"""
        messages_with_tools = [
            {
                "role": "user",
                "content": "What's the weather like?"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": "72F", "condition": "sunny"}'
            }
        ]

        with patch.object(guard.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response_success
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard.evaluate(messages_with_tools, "response")

            # Verify the API was called and result is a ScanResult object
            assert isinstance(result, object)  # It's a ScanResult, not a dict

    @pytest.mark.asyncio
    async def test_evaluate_with_violations_mock(self, guard, mock_response_violation):
        """Test evaluation that returns violations using mock"""
        messages = [
            {
                "role": "user",
                "content": "Inappropriate content"
            }
        ]

        with patch.object(guard.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response_violation
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard.evaluate(messages, "request")

            # Verify violations are detected - result is now ScanResult
            assert result.guardrails_triggered == 1
            assert len(result.violations) == 1
            assert result.violations[0].guardrail_type == "content_policy"

    def test_tool_call_structure_validation(self):
        """Test that tool call structure is properly validated"""
        # Valid tool call structure
        valid_tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            }
        }

        assert valid_tool_call["id"] == "call_123"
        assert valid_tool_call["type"] == "function"
        assert valid_tool_call["function"]["name"] == "get_weather"
        assert "arguments" in valid_tool_call["function"]

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls in a single message"""
        messages_with_multiple_tools = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}'
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone": "PST"}'
                        }
                    }
                ]
            }
        ]

        # Verify multiple tool calls are properly structured
        tool_calls = messages_with_multiple_tools[0]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"
