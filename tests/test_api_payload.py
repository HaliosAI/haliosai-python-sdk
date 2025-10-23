#!/usr/bin/env python3
"""
Unit tests for API payload handling with tool calls
"""
import pytest
from unittest.mock import patch, MagicMock
from haliosai import HaliosGuard


class TestApiPayload:
    """Unit tests for API payload handling"""

    def test_tool_calls_in_payload(self):
        """Test that tool calls are properly included in API payload"""
        guard = HaliosGuard(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com"
        )

        # Test conversation with tool calls
        messages_with_tools = [
            {
                "role": "user",
                "content": "Calculate the result of 15 * 8 + 42"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculate_math",
                            "arguments": '{"expression":"15 * 8 + 42"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"result": 162}'
            }
        ]

        # Test that extract_messages handles tool calls correctly
        extracted = guard.extract_messages(messages_with_tools)
        assert extracted == messages_with_tools

        # Test message counting
        assert len(extracted) == 3

        # Test that tool calls are preserved
        assistant_message = extracted[1]
        assert assistant_message["role"] == "assistant"
        assert assistant_message["content"] is None
        assert "tool_calls" in assistant_message
        assert len(assistant_message["tool_calls"]) == 1
        assert assistant_message["tool_calls"][0]["function"]["name"] == "calculate_math"

    def test_response_content_with_tool_calls(self):
        """Test response content extraction with tool calls"""
        guard = HaliosGuard(agent_id="test-agent")

        # Mock response with tool calls
        response_with_tools = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}'
                        }
                    }]
                }
            }]
        }

        content = guard.extract_response_content(response_with_tools)
        # Should create a description when there are tool calls but no content
        assert "Assistant called tools: get_weather" in content

    @pytest.mark.asyncio
    async def test_evaluate_with_tool_calls_mock(self):
        """Test evaluation with tool calls using mocked httpx"""
        guard = HaliosGuard(agent_id="test-agent", api_key="test-key")
        
        # Initialize HTTP client for testing
        guard._ensure_http_client_for_testing()

        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}
                }]
            }
        ]

        mock_response = {
            "guardrails_triggered": 0,
            "result": [],
            "request": {"message_count": 2, "content_length": 150}
        }

        with patch.object(guard.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard.evaluate(messages, "request")

            # result is now a ScanResult object, verify its attributes
            assert result.message_count == 2
            assert result.content_length == 150
            # Verify the API was called with the correct payload
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['messages'] == messages
            assert payload['invocation_type'] == "request"


# Legacy integration test (kept for reference but marked as integration)
@pytest.mark.integration
def test_tool_calls_integration():
    """Integration test for tool calls (requires real API)"""
    pytest.skip("Integration test - requires real HaliosAI API connection")


if __name__ == "__main__":
    pytest.main([__file__])
