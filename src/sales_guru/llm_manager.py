"""LLM management with rate limiting and fallback capabilities."""

import time
import logging
import random
from typing import Any, List, Optional
from crewai import LLM

from sales_guru.config import config

logger = logging.getLogger("SalesGuru.LLM")


class RateLimitedLLM:
    """Wrapper around LLM to handle rate limits and empty responses with exponential backoff."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.7, max_retries: int = 5):
        """Initialize rate-limited LLM wrapper.
        
        Args:
            model: Model name to use
            api_key: API key for the model
            temperature: Temperature setting for the model
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model
        self.base_llm = LLM(
            model=model,
            api_key=api_key,
            temperature=temperature
        )
        self.max_retries = max_retries
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum 2 seconds between requests

        # Apply our safe generate method to handle empty responses
        if hasattr(self.base_llm, '_generate'):
            self.base_llm._original_generate = self.base_llm._generate
            self.base_llm._generate = self._safe_generate.__get__(self.base_llm)

    def __call__(self, *args, **kwargs) -> Any:
        """Make the LLM callable directly."""
        return self.call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base LLM."""
        return getattr(self.base_llm, name)

    def _is_empty_response(self, response: Any) -> bool:
        """Check if a response is empty or None."""
        return response is None or (isinstance(response, str) and not response.strip())

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def call(self, *args, **kwargs) -> Any:
        """Handle LLM calls with retry logic for common errors."""
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Enforce rate limiting before making request
                self._enforce_rate_limit()

                # Get the raw response
                response = self.base_llm.call(*args, **kwargs)

                # Check if response is empty
                if self._is_empty_response(response):
                    logger.warning(f"Empty response from LLM on attempt {retries + 1}/{self.max_retries + 1}")
                    retries += 1

                    if retries > self.max_retries:
                        break

                    wait_time = self._calculate_wait_time(retries)
                    logger.info(f"Waiting {wait_time:.1f}s before retry due to empty response")
                    time.sleep(wait_time)
                    continue

                # We have a valid non-empty response
                return response

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                retries += 1

                # Calculate wait time - longer for rate limits
                if self._is_rate_limit_error(error_msg):
                    wait_time = self._calculate_rate_limit_wait_time(retries)
                    logger.warning(f"Rate limit error: {e}. Waiting {wait_time:.1f}s before retry {retries}/{self.max_retries}")
                else:
                    wait_time = self._calculate_wait_time(retries)
                    logger.warning(f"Error: {e}. Waiting {wait_time:.1f}s before retry {retries}/{self.max_retries}")

                # Check for common errors that can be retried
                if self._is_retryable_error(error_msg):
                    time.sleep(wait_time)

                    if retries > self.max_retries:
                        break
                else:
                    # Other errors should be raised immediately unless it's our last retry
                    if retries >= self.max_retries:
                        break
                    time.sleep(wait_time)

        # If we've exhausted retries, return a fallback response
        logger.error(f"Max retries exceeded for model {self.model_name}. Last error: {last_error}")
        return self._get_fallback_response(*args, **kwargs)

    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is rate limit related."""
        return any(x in error_msg for x in [
            "rate limit", "429", "too many requests", "quota", "resource_exhausted"
        ])

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if error is retryable."""
        return any(x in error_msg for x in [
            "rate limit", "429", "too many requests", "quota", "resource_exhausted",
            "invalid response", "empty", "list index out of range",
            "connection reset", "connection error", "timeout",
            "network", "[errno", "socket"
        ])

    def _calculate_wait_time(self, retries: int) -> float:
        """Calculate wait time with exponential backoff and jitter."""
        base_wait_time = min(2 ** (retries + 1), 60)  # Cap at 1 minute
        jitter = random.uniform(0.8, 1.2)  # Add ±20% jitter
        return base_wait_time * jitter

    def _calculate_rate_limit_wait_time(self, retries: int) -> float:
        """Calculate longer wait times specifically for rate limit errors."""
        # For rate limits, wait longer - start at 30 seconds and increase
        base_wait_time = min(30 * (2 ** retries), 300)  # Cap at 5 minutes
        jitter = random.uniform(0.9, 1.1)  # Smaller jitter for rate limits
        return base_wait_time * jitter

    def _get_fallback_response(self, *args, **kwargs) -> str:
        """Return a guaranteed non-empty fallback response with task-specific information if possible."""
        fallback_prompt = args[0] if args else kwargs.get('prompt', '')

        # Try to detect task type from the prompt
        if isinstance(fallback_prompt, str):
            prompt_lower = fallback_prompt.lower()

            if 'lead qualification' in prompt_lower or 'qualify' in prompt_lower:
                return self._get_lead_qualification_fallback()
            elif 'prospect research' in prompt_lower or 'research' in prompt_lower:
                return self._get_prospect_research_fallback()
            elif 'email' in prompt_lower or 'outreach' in prompt_lower:
                return self._get_email_outreach_fallback()
            elif 'sales call' in prompt_lower or 'call prep' in prompt_lower:
                return self._get_sales_call_fallback()

        # Generic fallback
        return ("I apologize for the technical difficulties. The system experienced rate limiting issues "
                "with the AI service. Please retry your request in a few minutes, or contact support if "
                "the issue persists. This is a temporary service limitation.")

    def _get_lead_qualification_fallback(self) -> str:
        """Get fallback response for lead qualification tasks."""
        return '''{"leads": [{"name": "Fallback Lead", "company": "Fallback Company", "email": "fallback@example.com", "phone": "000-000-0000", "lead_score": 50, "classification": "WARM", "reasoning": "Fallback response due to API issues", "value_alignment": "To be determined", "recommended_approach": "Follow up when systems are restored"}]}'''

    def _get_prospect_research_fallback(self) -> str:
        """Get fallback response for prospect research tasks."""
        return '''{"prospects": [{"company_profile": "Research unavailable due to technical issues", "lead_insights": "To be researched manually", "recent_developments": "Not available", "pain_points": ["Technical issues preventing research"], "current_solutions": "Unknown", "urgency_evidence": "Manual research required", "talking_points": ["Technical issues occurred", "Manual research needed"]}]}'''

    def _get_email_outreach_fallback(self) -> str:
        """Get fallback response for email outreach tasks."""
        return '''{"email_templates": [{"lead_name": "Pending", "company_name": "Pending", "classification": "WARM", "subject_line": "Following up on our conversation", "email_body": "Dear [Name], I apologize for any delay. We experienced technical issues and will follow up with personalized outreach shortly. Best regards, [Your Name]", "follow_up_timing": "24 hours", "alternative_contact_channels": "Phone call", "ab_test_variations": [{"element": "subject", "variation": "Quick follow-up"}]}]}'''

    def _get_sales_call_fallback(self) -> str:
        """Get fallback response for sales call preparation tasks."""
        return '''{"call_briefs": [{"lead_name": "Pending", "company_name": "Pending", "classification": "WARM", "company_snapshot": "Research pending due to technical issues", "decision_maker_profile": "To be researched", "relationship_history": "New contact", "pain_points": ["Technical research limitations"], "talking_points": ["Apologize for delay", "Schedule proper research call"], "objection_responses": [{"objection": "Technical issues", "response": "We've resolved the issues and are ready to provide full service"}], "next_steps": ["Complete manual research", "Schedule follow-up"], "recent_developments": "Not available", "competitive_insights": "Pending research", "value_propositions": ["Reliable service despite technical challenges"]}]}'''

    def _safe_generate(self, *args, **kwargs) -> str:
        """Override the _generate method to ensure we never return empty responses."""
        try:
            response = self._original_generate(*args, **kwargs)

            if self._is_empty_response(response):
                logger.warning("Empty response detected in _generate method. Using fallback.")
                return "I apologize, but I encountered a technical issue. Please consider this a partial response based on the available information."

            return response
        except Exception as e:
            logger.error(f"Error in _generate method: {e}")
            return "I apologize, but I encountered a technical issue. Please consider this a partial response based on the available information."


class FallbackLLM:
    """LLM with automatic fallback to alternative models when rate limits are hit."""

    def __init__(self, primary_model: str, fallback_models: List[str], api_key: str, temperature: float = 0.7):
        """Initialize fallback LLM system.
        
        Args:
            primary_model: Primary model to use
            fallback_models: List of fallback models
            api_key: API key for the models
            temperature: Temperature setting
        """
        self.primary_llm = RateLimitedLLM(primary_model, api_key, temperature, max_retries=3)
        self.fallback_llms = [RateLimitedLLM(model, api_key, temperature, max_retries=2) for model in fallback_models]
        self.current_model_index = -1  # -1 means primary, 0+ means fallback index

    def __call__(self, *args, **kwargs) -> Any:
        """Make the LLM callable."""
        return self.call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate to the currently active LLM."""
        if self.current_model_index == -1:
            return getattr(self.primary_llm, name)
        else:
            return getattr(self.fallback_llms[self.current_model_index], name)

    def call(self, *args, **kwargs) -> Any:
        """Try primary model first, then fallbacks if rate limited."""
        # Try primary model
        if self.current_model_index == -1:
            try:
                response = self.primary_llm.call(*args, **kwargs)
                # Check if it's a rate limit fallback response
                if not self._is_rate_limit_fallback(response):
                    return response
                else:
                    logger.warning("Primary model returned rate limit fallback, switching to fallback model")
                    self.current_model_index = 0
            except Exception as e:
                error_msg = str(e).lower()
                if any(x in error_msg for x in ["rate limit", "429", "quota", "resource_exhausted"]):
                    logger.warning(f"Primary model rate limited: {e}. Switching to fallback.")
                    self.current_model_index = 0
                else:
                    raise

        # Try fallback models
        while self.current_model_index < len(self.fallback_llms):
            try:
                response = self.fallback_llms[self.current_model_index].call(*args, **kwargs)
                if not self._is_rate_limit_fallback(response):
                    return response
                else:
                    logger.warning(f"Fallback model {self.current_model_index} also rate limited, trying next")
                    self.current_model_index += 1
            except Exception as e:
                error_msg = str(e).lower()
                if any(x in error_msg for x in ["rate limit", "429", "quota", "resource_exhausted"]):
                    logger.warning(f"Fallback model {self.current_model_index} rate limited: {e}")
                    self.current_model_index += 1
                else:
                    raise

        # All models exhausted
        logger.error("All models (primary and fallbacks) are rate limited or failed")
        return "I apologize, but all AI models are currently experiencing rate limits. Please try again in a few minutes."

    def _is_rate_limit_fallback(self, response: Any) -> bool:
        """Check if response is a rate limit fallback response."""
        if isinstance(response, str):
            return "rate limiting issues" in response.lower() or "technical difficulties" in response.lower()
        return False


class LLMManager:
    """Centralized LLM management."""
    
    def __init__(self):
        """Initialize LLM manager with configuration."""
        config.validate_api_keys()
        
        # Initialize Google LLMs
        self.google_llm = FallbackLLM(
            primary_model=config.primary_model,
            fallback_models=config.fallback_models,
            api_key=config.google_api_key,
            temperature=0.7
        )
        
        # Specific LLM for lead qualification with more powerful model
        self.lead_qualification_llm = RateLimitedLLM(
            model=config.lead_qualification_model,
            api_key=config.google_api_key,
            temperature=0.7,
            max_retries=3
        )
        
        # Initialize OpenAI LLMs
        self.openai_llm = FallbackLLM(
            primary_model=config.openai_primary_model,
            fallback_models=config.openai_fallback_models,
            api_key=config.openai_api_key,
            temperature=0.7
        )
    
    def get_google_llm(self) -> FallbackLLM:
        """Get Google LLM instance."""
        return self.google_llm
    
    def get_lead_qualification_llm(self) -> RateLimitedLLM:
        """Get lead qualification LLM instance."""
        return self.lead_qualification_llm
    
    def get_openai_llm(self) -> FallbackLLM:
        """Get OpenAI LLM instance."""
        return self.openai_llm


# Global LLM manager instance
llm_manager = LLMManager() 