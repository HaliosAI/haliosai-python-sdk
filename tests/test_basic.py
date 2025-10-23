"""
Basic tests for HaliosAI SDK
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from haliosai import HaliosGuard, ExecutionResult, ViolationAction


class TestHaliosGuard:
    """Test cases for HaliosGuard class"""
    
    def test_init(self):
        """Test HaliosGuard initialization"""
        guard_instance = HaliosGuard(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com"
        )
        
        assert guard_instance.agent_id == "test-agent"
        assert guard_instance.api_key == "test-key"
        assert guard_instance.base_url == "http://test.com"
        assert guard_instance.parallel is False
    
    def test_extract_messages_from_kwargs(self):
        """Test message extraction from kwargs"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        messages = [{"role": "user", "content": "test"}]
        extracted = guard_instance.extract_messages(messages=messages)
        
        assert extracted == messages
    
    def test_extract_messages_from_args(self):
        """Test message extraction from positional args"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        messages = [{"role": "user", "content": "test"}]
        extracted = guard_instance.extract_messages(messages)
        
        assert extracted == messages
    
    def test_extract_messages_from_string(self):
        """Test message extraction from string argument"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        extracted = guard_instance.extract_messages("Hello world")
        expected = [{"role": "user", "content": "Hello world"}]
        
        assert extracted == expected
    
    def test_extract_response_content_string(self):
        """Test response content extraction from string"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        content = guard_instance.extract_response_content("test response")
        assert content == "test response"
    
    def test_extract_response_content_dict(self):
        """Test response content extraction from dict"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        response = {
            "choices": [{"message": {"content": "test response"}}]
        }
        content = guard_instance.extract_response_content(response)
        assert content == "test response"
    
    @pytest.mark.asyncio
    async def test_check_violations_none(self):
        """Test violation checking with no violations"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        result = {"guardrails_triggered": 0, "result": []}
        action, violations = await guard_instance.check_violations(result)
        
        assert len(violations) == 0
        assert action == ViolationAction.PASS
    
    @pytest.mark.asyncio
    async def test_check_violations_triggered(self):
        """Test violation checking with violations"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        result = {
            "guardrails_triggered": 1,
            "result": [
                {
                    "triggered": True,
                    "guardrail_type": "content_safety",
                    "analysis": {"explanation": "harmful content detected"}
                }
            ]
        }
        action, violations = await guard_instance.check_violations(result)
        
        assert len(violations) > 0
        assert action == ViolationAction.BLOCK
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful guardrail evaluation"""
        guard_instance = HaliosGuard(agent_id="test-agent", api_key="test-key")
        
        guard_instance = HaliosGuard(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com"
        )
        
        # Initialize HTTP client for testing
        guard_instance._ensure_http_client_for_testing()
        
        mock_response = {
            "guardrails_triggered": 0,
            "result": [],
            "request": {"message_count": 2, "content_length": 100}
        }
        
        with patch.object(guard_instance.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard_instance.evaluate([{"role": "user", "content": "test"}], "request")
            
            # result is now a ScanResult object
            assert isinstance(result, object)  # Check it's an object, not a dict
            assert result.message_count == 2
            assert result.content_length == 100
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_with_tool_calls(self):
        """Test evaluation with tool calls in messages"""
        guard_instance = HaliosGuard(agent_id="test-agent", api_key="test-key")
        
        # Initialize HTTP client for testing
        guard_instance._ensure_http_client_for_testing()
        
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
        
        with patch.object(guard_instance.http_client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None

            mock_post.return_value = mock_response_obj

            result = await guard_instance.evaluate(messages, "request")

            # result is now a ScanResult object
            assert result.message_count == 2
            assert result.content_length == 150
            # Verify tool calls were included in the payload
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['messages'] == messages


class TestEnvironmentVariables:
    """Test environment variable handling"""
    
    @patch.dict('os.environ', {'HALIOS_API_KEY': 'env-key'})
    def test_api_key_from_env(self):
        """Test API key loading from environment"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        assert guard_instance.api_key == "env-key"
    
    @patch.dict('os.environ', {'HALIOS_BASE_URL': 'http://env.com'})
    def test_base_url_from_env(self):
        """Test base URL loading from environment"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        assert guard_instance.base_url == "http://env.com"


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_evaluate_http_error(self):
        """Test HTTP error handling"""
        guard_instance = HaliosGuard(agent_id="test-agent", api_key="test-key")
        
        # Initialize HTTP client for testing
        guard_instance._ensure_http_client_for_testing()
        
        with patch.object(guard_instance.http_client, 'post') as mock_post:
            mock_post.side_effect = Exception("HTTP 500")

            with pytest.raises(Exception):
                await guard_instance.evaluate([{"role": "user", "content": "test"}], "request")
    
    def test_extract_messages_empty(self):
        """Test message extraction with no valid messages"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        extracted = guard_instance.extract_messages()
        assert extracted == []
    
    def test_extract_response_content_invalid(self):
        """Test response content extraction from invalid format"""
        guard_instance = HaliosGuard(agent_id="test-agent")
        
        content = guard_instance.extract_response_content(123)
        assert content == "123"  # Should convert to string


# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
