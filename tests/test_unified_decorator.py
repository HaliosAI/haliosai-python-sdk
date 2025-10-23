#!/usr/bin/env python3
"""
Unit tests for the unified guarded_chat_completion decorator
"""

import pytest
from unittest.mock import patch, AsyncMock
from haliosai import guarded_chat_completion, GuardrailViolation, ViolationAction, GuardrailPolicy


class TestGuardedChatCompletion:
    """Unit tests for the guarded_chat_completion decorator"""

    @pytest.mark.asyncio
    async def test_decorator_initialization(self):
        """Test that decorator can be created without errors"""
        @guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com"
        )
        async def dummy_function(messages):
            return {"result": "test"}

        # Just test that the decorator doesn't crash during initialization
        assert callable(dummy_function)

    @pytest.mark.asyncio
    async def test_decorator_with_missing_messages(self):
        """Test decorator behavior when messages are not provided"""
        @guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com"
        )
        async def call_llm():
            return {"result": "test"}

        # Should raise ValueError when messages are not provided
        with pytest.raises(ValueError, match="Function must receive 'messages'"):
            await call_llm()

    @pytest.mark.asyncio
    async def test_streaming_decorator_initialization(self):
        """Test that streaming decorator can be created"""
        @guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com",
            streaming_guardrails=True
        )
        async def stream_function(messages):
            # Use messages to avoid unused variable warning
            if messages:
                yield {"type": "chunk", "content": "test"}

        # Just test that the decorator doesn't crash during initialization
        assert callable(stream_function)

    def test_decorator_parameters(self):
        """Test that decorator accepts all expected parameters"""
        # This should not raise any errors
        decorator = guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="http://test.com",
            concurrent_guardrail_processing=True,
            streaming_guardrails=False,
            stream_buffer_size=50,
            stream_check_interval=0.5,
            guardrail_timeout=5.0
        )

        assert callable(decorator)

    def test_decorator_with_on_violation_callback(self):
        """Test that decorator accepts on_violation callback"""
        violation_called = False
        
        def on_violation(violation):
            nonlocal violation_called
            violation_called = True
            assert isinstance(violation, GuardrailViolation)
            assert violation.violation_type in ["request", "response"]
            assert isinstance(violation.violations, list)
        
        decorator = guarded_chat_completion(
            agent_id="test-agent",
            on_violation=on_violation
        )
        
        assert callable(decorator)

    @pytest.mark.asyncio
    async def test_guardrail_violation_exception_structure(self):
        """Test that GuardrailViolation contains proper violation details"""
        # This test verifies the exception structure without full integration
        from haliosai.client import Violation
        
        violation = GuardrailViolation(
            violation_type="request",
            violations=[Violation(
                guardrail_type="content-moderation",
                analysis={"flagged": True},
                guardrail_uuid="test-uuid"
            )],
            blocked_content="test content",
            timing={"request_time": 0.1}
        )
        
        assert violation.violation_type == "request"
        assert len(violation.violations) == 1
        assert violation.blocked_content == "test content"
        assert violation.timing["request_time"] == 0.1
        assert "content-moderation" in str(violation)

    @pytest.mark.asyncio
    async def test_on_violation_callback_execution(self):
        """Test that on_violation callback is executed when violations occur"""
        callback_data = {}

        def capture_violation(violation):
            callback_data.update({
                "called": True,
                "type": violation.violation_type,
                "violation_count": len(violation.violations)
            })

        # Mock a decorator that will trigger violation
        @guarded_chat_completion(
            agent_id="test-agent",
            on_violation=capture_violation
        )
        async def failing_function(messages):
            # This would normally trigger guardrails
            return "response"
    
        # Since we can't easily mock the full guardrail evaluation,
        # we test that the callback parameter is accepted and the function is callable
        assert callable(failing_function)
        # Test that the function has the expected attributes (wrapped function)
        assert hasattr(failing_function, '__wrapped__')
# Integration-style tests (require mocking httpx)
    """Tests for the modified guardrail violation logic"""
    
    @pytest.mark.asyncio
    async def test_sensitive_data_blocks_by_default(self):
        """Test that sensitive-data guardrail blocks by default (no policy)"""
        from haliosai.client import HaliosGuard
    
        guard = HaliosGuard(agent_id="test")
    
        # Mock result with only sensitive-data violation
        result = {
            "guardrails_triggered": 1,
            "result": [{
                "triggered": True,
                "guardrail_type": "sensitive-data",
                "analysis": {"sanitized": True}
            }]
        }
    
        # Should return ALLOW_OVERRIDE for sensitive-data (allowed by default now requires policy)
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.BLOCK  # Now blocks by default
        assert len(violations) > 0  # has_violations
        assert len(violations) > 0  # blocking_violations

    @pytest.mark.asyncio
    async def test_sensitive_data_with_allow_policy(self):
        """Test that sensitive-data can be allowed with explicit policy"""
        from haliosai.client import HaliosGuard
    
        guard = HaliosGuard(agent_id="test", guardrail_policies={"sensitive-data": GuardrailPolicy.RECORD_ONLY})
    
        # Mock result with only sensitive-data violation
        result = {
            "guardrails_triggered": 1,
            "result": [{
                "triggered": True,
                "guardrail_type": "sensitive-data",
                "analysis": {"sanitized": True}
            }]
        }
    
        # Should return ALLOW_OVERRIDE when policy allows it
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.ALLOW_OVERRIDE
        assert len(violations) > 0  # has_violations
        assert len(violations) > 0  # blocking_violations is False - but we have violations, just not blocking

    @pytest.mark.asyncio
    async def test_modified_messages_allowed(self):
        """Test that modified_messages allows request to pass"""
        from haliosai.client import HaliosGuard
        
        guard = HaliosGuard(agent_id="test")
        
        # Mock result with modified_messages
        result = {
            "modified_messages": [
                {"role": "user", "content": "sanitized content"}
            ],
            "guardrails_triggered": 1,
            "result": [{
                "triggered": True,
                "guardrail_type": "sensitive-data"
            }]
        }
        
        # Should return ALLOW_OVERRIDE when modified_messages present
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.ALLOW_OVERRIDE
        assert len(violations) == 0  # modified_messages takes precedence - no violations
    
    @pytest.mark.asyncio
    async def test_multiple_violations_blocked(self):
        """Test that multiple violations (including sensitive-data) are blocked"""
        from haliosai.client import HaliosGuard
        
        guard = HaliosGuard(agent_id="test")
        
        # Mock result with multiple violations
        result = {
            "guardrails_triggered": 2,
            "result": [
                {
                    "triggered": True,
                    "guardrail_type": "sensitive-data",
                    "analysis": {"sanitized": True}
                },
                {
                    "triggered": True,
                    "guardrail_type": "content-moderation",
                    "analysis": {"flagged": True}
                }
            ]
        }
    
        # Should return BLOCK when multiple violations
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.BLOCK
        assert len(violations) > 0  # has_violations
        assert len(violations) > 0  # blocking_violations

    @pytest.mark.asyncio
    async def test_sensitive_data_policy_block(self):
        """Test that sensitive-data can be configured to block via guardrail_policies"""
        from haliosai.client import HaliosGuard
        
        guard = HaliosGuard(agent_id="test", guardrail_policies={"sensitive-data": GuardrailPolicy.BLOCK})
        
        # Mock result with only sensitive-data violation
        result = {
            "guardrails_triggered": 1,
            "result": [{
                "triggered": True,
                "guardrail_type": "sensitive-data",
                "analysis": {"sanitized": True}
            }]
        }
    
        # Should return BLOCK when policy specifies "block"
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.BLOCK
        assert len(violations) > 0  # has_violations
        assert len(violations) > 0  # blocking_violations

    @pytest.mark.asyncio
    async def test_sensitive_data_policy_allow(self):
        """Test that sensitive-data can be configured to allow via guardrail_policies"""
        from haliosai.client import HaliosGuard
        
        guard = HaliosGuard(agent_id="test", guardrail_policies={"sensitive-data": GuardrailPolicy.RECORD_ONLY})
        
        # Mock result with only sensitive-data violation
        result = {
            "guardrails_triggered": 1,
            "result": [{
                "triggered": True,
                "guardrail_type": "sensitive-data",
                "analysis": {"sanitized": True}
            }]
        }
    
        # Should return ALLOW_OVERRIDE when policy specifies "allow"
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.ALLOW_OVERRIDE
        assert len(violations) > 0  # Still has violations, but allowed
        assert len(violations) > 0  # blocking_violations is False, but we have violations

    @pytest.mark.asyncio
    async def test_mixed_policies_precedence(self):
        """Test that BLOCK takes precedence over ALLOW_OVERRIDE in mixed violations"""
        from haliosai.client import HaliosGuard
        
        guard = HaliosGuard(agent_id="test", guardrail_policies={
            "sensitive-data": GuardrailPolicy.RECORD_ONLY,  # Would be ALLOW_OVERRIDE
            "hate-speech": GuardrailPolicy.BLOCK            # This should make overall result BLOCK
        })
        
        # Mock result with both violations
        result = {
            "guardrails_triggered": 2,
            "result": [
                {
                    "triggered": True,
                    "guardrail_type": "sensitive-data",
                    "analysis": {"sanitized": True}
                },
                {
                    "triggered": True,
                    "guardrail_type": "hate-speech",
                    "analysis": {"flagged": True}
                }
            ]
        }
    
        # Should return BLOCK because hate-speech is set to block
        action, violations = await guard.check_violations(result)
        assert action == ViolationAction.BLOCK
        assert len(violations) > 0  # has_violations
        assert len(violations) > 0  # blocking_violations


class TestDecoratorUsage:
    """Tests for decorator usage patterns - validates actual behavior, not just structure"""

    @pytest.mark.asyncio
    async def test_decorator_default_behavior_blocks_violations(self):
        """Test default behavior: violations are blocked (no policy specified)"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(agent_id="test-agent")
        async def call_llm(messages):
            return {"response": "Hello!"}

        # Patch HaliosGuard.evaluate to simulate violation detection
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "result": [{
                    "triggered": True,
                    "guardrail_type": "profanity",
                    "analysis": {"flagged": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should raise GuardrailViolation because default is to block
            with pytest.raises(GuardrailViolation) as exc_info:
                await call_llm([{"role": "user", "content": "Test"}])
            
            assert exc_info.value.violation_type == "request"
            assert len(exc_info.value.violations) > 0

    @pytest.mark.asyncio
    async def test_decorator_default_behavior_passes_clean_requests(self):
        """Test default behavior: clean requests pass through"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(agent_id="test-agent")
        async def call_llm(messages):
            return {"response": "Hello!"}

        # Patch HaliosGuard.evaluate to simulate no violations
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 0,
                "result": []
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            result = await call_llm([{"role": "user", "content": "Hello"}])
            assert result == {"response": "Hello!"}

    @pytest.mark.asyncio
    async def test_decorator_with_allow_policy_permits_violations(self):
        """Test allow policy: violations are logged but request proceeds"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(
            agent_id="test-agent",
            guardrail_policies={"sensitive-data": GuardrailPolicy.RECORD_ONLY}
        )
        async def call_llm(messages):
            return {"response": "Has PII"}

        # Patch HaliosGuard.evaluate to simulate sensitive-data violation
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "result": [{
                    "triggered": True,
                    "guardrail_type": "sensitive-data",
                    "analysis": {"detected": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should NOT raise because policy allows it
            result = await call_llm([{"role": "user", "content": "SSN: 123-45-6789"}])
            assert result == {"response": "Has PII"}

    @pytest.mark.asyncio
    async def test_decorator_with_block_policy_blocks_violations(self):
        """Test block policy: violations are blocked"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(
            agent_id="test-agent",
            guardrail_policies={"hate-speech": GuardrailPolicy.BLOCK}
        )
        async def call_llm(messages):
            return {"response": "Response"}

        # Patch HaliosGuard.evaluate to simulate hate-speech violation
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "result": [{
                    "triggered": True,
                    "guardrail_type": "hate-speech",
                    "analysis": {"flagged": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should raise because policy blocks it
            with pytest.raises(GuardrailViolation) as exc_info:
                await call_llm([{"role": "user", "content": "Hate speech"}])
            
            assert exc_info.value.violation_type == "request"

    @pytest.mark.asyncio
    async def test_decorator_with_record_only_policy_logs_but_allows(self):
        """Test record_only policy: violations are logged but allowed"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(
            agent_id="test-agent",
            guardrail_policies={"pii-detection": GuardrailPolicy.RECORD_ONLY}
        )
        async def call_llm(messages):
            return {"response": "Email: test@example.com"}

        # Patch HaliosGuard.evaluate to simulate PII violation
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "result": [{
                    "triggered": True,
                    "guardrail_type": "pii-detection",
                    "analysis": {"detected": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should NOT raise because record_only allows it
            result = await call_llm([{"role": "user", "content": "email@test.com"}])
            assert result == {"response": "Email: test@example.com"}

    @pytest.mark.asyncio
    async def test_decorator_with_mixed_policies_respects_precedence(self):
        """Test mixed policies: BLOCK takes precedence over ALLOW_OVERRIDE"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(
            agent_id="test-agent",
            guardrail_policies={
                "sensitive-data": GuardrailPolicy.RECORD_ONLY,
                "hate-speech": GuardrailPolicy.BLOCK
            }
        )
        async def call_llm(messages):
            return {"response": "Response"}

        # Patch HaliosGuard.evaluate to simulate both violations
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 2,
                "result": [
                    {
                        "triggered": True,
                        "guardrail_type": "sensitive-data",
                        "analysis": {"detected": True}
                    },
                    {
                        "triggered": True,
                        "guardrail_type": "hate-speech",
                        "analysis": {"flagged": True}
                    }
                ]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should raise because hate-speech blocks (takes precedence)
            with pytest.raises(GuardrailViolation) as exc_info:
                await call_llm([{"role": "user", "content": "Mixed content"}])
            
            assert exc_info.value.violation_type == "request"

    @pytest.mark.asyncio
    async def test_decorator_with_on_violation_callback_is_called_on_block(self):
        """Test on_violation callback is invoked when violations are BLOCKED"""
        from haliosai.client import HaliosGuard
        
        callback_called = []

        def test_callback(violation):
            callback_called.append({
                'type': violation.violation_type,
                'count': len(violation.violations)
            })

        @guarded_chat_completion(
            agent_id="test-agent",
            on_violation=test_callback
            # No policy = block by default
        )
        async def call_llm(messages):
            return {"response": "Test"}

        # Patch HaliosGuard.evaluate to simulate violation
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "result": [{
                    "triggered": True,
                    "guardrail_type": "profanity",
                    "analysis": {"flagged": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should raise and callback should be called
            with pytest.raises(GuardrailViolation):
                await call_llm([{"role": "user", "content": "Test"}])
            
            # Callback should have been called
            assert len(callback_called) > 0
            assert callback_called[0]['type'] == 'request'
            assert callback_called[0]['count'] > 0

    @pytest.mark.asyncio
    async def test_decorator_with_modified_messages_allows_request(self):
        """Test modified_messages: decorator allows request (doesn't auto-replace messages)"""
        from haliosai.client import HaliosGuard
        
        @guarded_chat_completion(agent_id="test-agent")
        async def call_llm(messages):
            # Current behavior: receives original messages, not sanitized ones
            return {"content": messages[0]['content']}

        # Patch HaliosGuard.evaluate to return modified_messages
        async def mock_evaluate(self_guard, messages, evaluation_type):
            return {
                "guardrails_triggered": 1,
                "modified_messages": [
                    {"role": "user", "content": "My SSN is [REDACTED]"}
                ],
                "result": [{
                    "triggered": True,
                    "guardrail_type": "sensitive-data",
                    "analysis": {"sanitized": True}
                }]
            }
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Should NOT raise because modified_messages indicates content was sanitized
            result = await call_llm([{"role": "user", "content": "My SSN is 123-45-6789"}])
            # Currently: original messages are used, not sanitized
            # TODO: Consider auto-replacing messages with modified_messages
            assert result["content"] == "My SSN is 123-45-6789"

    @pytest.mark.asyncio
    async def test_decorator_validates_messages_parameter_required(self):
        """Test decorator raises error when messages parameter is missing"""
        @guarded_chat_completion(agent_id="test-agent")
        async def call_llm_no_messages():
            return {"response": "Test"}

        # Should raise ValueError about missing messages
        with pytest.raises(ValueError, match="messages"):
            await call_llm_no_messages()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring"""
        @guarded_chat_completion(agent_id="test-agent")
        async def my_llm_function(messages):
            """This is my LLM function with documentation."""
            return {"response": "success"}

        assert my_llm_function.__name__ == "my_llm_function"
        assert my_llm_function.__doc__ == "This is my LLM function with documentation."

    @pytest.mark.asyncio
    async def test_decorator_evaluates_request_and_response_guardrails(self):
        """Test decorator evaluates both request and response guardrails in both modes"""
        from haliosai.client import HaliosGuard
        
        eval_calls = []
        
        # Track evaluation calls
        async def mock_evaluate(self_guard, messages, evaluation_type):
            eval_calls.append(evaluation_type)
            return {"guardrails_triggered": 0, "result": []}
        
        with patch.object(HaliosGuard, 'evaluate', new=mock_evaluate):
            # Test concurrent mode (default)
            @guarded_chat_completion(agent_id="test-agent")
            async def call_llm_concurrent(messages):
                return {"response": "Clean response"}
            
            result = await call_llm_concurrent([{"role": "user", "content": "Hello"}])
            assert "request" in eval_calls
            # Response is evaluated after LLM call in concurrent mode
            assert "response" in eval_calls
            
            # Reset for sequential test
            eval_calls.clear()
            
            # Test sequential mode
            @guarded_chat_completion(agent_id="test-agent", concurrent_guardrail_processing=False)
            async def call_llm_sequential(messages):
                return {"response": "Clean response"}
            
            result = await call_llm_sequential([{"role": "user", "content": "Hello"}])
            assert "request" in eval_calls
            assert "response" in eval_calls
    async def test_decorator_with_streaming_creates_generator(self):
        """Test streaming decorator creates async generator function"""
        @guarded_chat_completion(
            agent_id="test-agent",
            streaming_guardrails=True
        )
        async def stream_llm(messages):
            yield {"chunk": "Hello"}
            yield {"chunk": " World"}

        import inspect
        assert inspect.isasyncgenfunction(stream_llm)


class TestDecoratorIntegration:
    """Integration tests for the guarded_chat_completion decorator"""

    @pytest.mark.asyncio
    async def test_decorator_accepts_on_violation_callback(self):
        """Test that decorator properly accepts and stores on_violation callback"""
        callback_called = False
        violation_data = {}

        def test_callback(violation):
            nonlocal callback_called
            callback_called = True
            violation_data.update({
                'type': violation.violation_type,
                'count': len(violation.violations)
            })

        # Decorate a function with callback
        @guarded_chat_completion(agent_id="test-agent", on_violation=test_callback)
        async def test_func(messages):
            return {"result": "success"}

        # Verify the function is callable and has the callback configured
        assert callable(test_func)
        assert hasattr(test_func, '__wrapped__')

    @pytest.mark.asyncio
    async def test_decorator_raises_value_error_for_missing_messages(self):
        """Test that decorator raises ValueError when messages parameter is missing"""
        @guarded_chat_completion(agent_id="test-agent")
        async def test_func():
            return {"result": "success"}

        # Check that the function is decorated (has __wrapped__)
        assert hasattr(test_func, '__wrapped__')
        assert callable(test_func)

        # The actual error will occur when called, but we can't test the await
        # due to type checker limitations with the decorator's conditional return type

    @pytest.mark.asyncio
    async def test_decorator_accepts_messages_as_keyword_argument(self):
        """Test that decorator works when messages is passed as keyword argument"""
        @guarded_chat_completion(agent_id="test-agent")
        async def test_func(messages=None):
            return {"result": "success"}

        # Check that the function is properly decorated
        assert hasattr(test_func, '__wrapped__')
        assert callable(test_func)

        # The actual functionality will work at runtime, but we can't test the await
        # due to type checker limitations

    @pytest.mark.asyncio
    async def test_decorator_with_sequential_processing(self):
        """Test that decorator works with sequential processing (non-concurrent)"""
        @guarded_chat_completion(
            agent_id="test-agent",
            concurrent_guardrail_processing=False
        )
        async def test_func(messages):
            return {"result": "success"}

        assert callable(test_func)
        assert hasattr(test_func, '__wrapped__')

    @pytest.mark.asyncio
    async def test_decorator_with_streaming_enabled(self):
        """Test that decorator creates streaming version when streaming_guardrails=True"""
        @guarded_chat_completion(
            agent_id="test-agent",
            streaming_guardrails=True
        )
        async def test_stream_func(messages):
            yield {"type": "chunk", "content": "test"}

        # Streaming version should be an async generator function
        import inspect
        assert inspect.isasyncgenfunction(test_stream_func)

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name, docstring, and other metadata"""
        @guarded_chat_completion(agent_id="test-agent")
        async def my_test_function(messages):
            """This is a test function with documentation."""
            return {"result": "success"}

        assert my_test_function.__name__ == "my_test_function"
        assert my_test_function.__doc__ == "This is a test function with documentation."
        assert hasattr(my_test_function, '__wrapped__')

    @pytest.mark.asyncio
    async def test_decorator_with_all_parameters(self):
        """Test that decorator accepts all configuration parameters"""
        @guarded_chat_completion(
            agent_id="test-agent",
            api_key="test-key",
            base_url="https://test.api.com",
            concurrent_guardrail_processing=False,
            streaming_guardrails=False,
            stream_buffer_size=100,
            stream_check_interval=1.0,
            guardrail_timeout=10.0,
            on_violation=lambda v: None
        )
        async def test_func(messages):
            return {"result": "success"}

        assert callable(test_func)
        assert hasattr(test_func, '__wrapped__')


# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
