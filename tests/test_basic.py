"""
Basic tests for HaliosAI SDK
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from haliosai import HaliosGuard, guard, ExecutionResult


class TestHaliosGuard:
    """Test cases for HaliosGuard class"""
    
    def test_init(self):
        """Test HaliosGuard initialization"""
        guard_instance = HaliosGuard(
            app_id="test-app", 
            api_key="test-key", 
            base_url="http://test.com"
        )
        
        assert guard_instance.app_id == "test-app"
        assert guard_instance.api_key == "test-key"
        assert guard_instance.base_url == "http://test.com"
        assert guard_instance.parallel is False
    
    def test_guard_factory(self):
        """Test guard factory function"""
        guard_instance = guard(app_id="test-app")
        
        assert isinstance(guard_instance, HaliosGuard)
        assert guard_instance.app_id == "test-app"
    
    def test_extract_messages_from_kwargs(self):
        """Test message extraction from kwargs"""
        guard_instance = HaliosGuard(app_id="test-app")
        
        messages = [{"role": "user", "content": "test"}]
        extracted = guard_instance.extract_messages(messages=messages)
        
        assert extracted == messages
    
    def test_extract_messages_from_args(self):
        """Test message extraction from positional args"""
        guard_instance = HaliosGuard(app_id="test-app")
        
        messages = [{"role": "user", "content": "test"}]
        extracted = guard_instance.extract_messages(messages)
        
        assert extracted == messages
    
    def test_extract_response_content_string(self):
        """Test response content extraction from string"""
        guard_instance = HaliosGuard(app_id="test-app")
        
        content = guard_instance.extract_response_content("test response")
        assert content == "test response"
    
    def test_extract_response_content_dict(self):
        """Test response content extraction from dict"""
        guard_instance = HaliosGuard(app_id="test-app")
        
        response = {
            "choices": [{"message": {"content": "test response"}}]
        }
        content = guard_instance.extract_response_content(response)
        assert content == "test response"
    
    @pytest.mark.asyncio
    async def test_check_violations_none(self):
        """Test violation checking with no violations"""
        guard_instance = HaliosGuard(app_id="test-app")
        
        result = {"guardrails_triggered": 0, "result": []}
        has_violations = await guard_instance.check_violations(result)
        
        assert has_violations is False
    
    @pytest.mark.asyncio
    async def test_check_violations_triggered(self):
        """Test violation checking with violations"""
        guard_instance = HaliosGuard(app_id="test-app")
        
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
        has_violations = await guard_instance.check_violations(result)
        
        assert has_violations is True


class TestEnvironmentVariables:
    """Test environment variable handling"""
    
    @patch.dict('os.environ', {'HALIOS_API_KEY': 'env-key'})
    def test_api_key_from_env(self):
        """Test API key loading from environment"""
        guard_instance = HaliosGuard(app_id="test-app")
        assert guard_instance.api_key == "env-key"
    
    @patch.dict('os.environ', {'HALIOS_BASE_URL': 'http://env.com'})
    def test_base_url_from_env(self):
        """Test base URL loading from environment"""
        guard_instance = HaliosGuard(app_id="test-app")
        assert guard_instance.base_url == "http://env.com"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
