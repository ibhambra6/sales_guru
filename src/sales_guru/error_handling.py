"""Centralized error handling for Sales Guru."""

import time
import random
import logging
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger("SalesGuru.ErrorHandling")


class SalesGuruError(Exception):
    """Base exception for Sales Guru application."""
    pass


class APIError(SalesGuruError):
    """Exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Exception for rate limiting errors."""
    pass


class NetworkError(SalesGuruError):
    """Exception for network connectivity errors."""
    pass


class ConfigurationError(SalesGuruError):
    """Exception for configuration-related errors."""
    pass


class ErrorClassifier:
    """Classify errors into different categories for appropriate handling."""
    
    @staticmethod
    def is_network_error(error_msg: str) -> bool:
        """Check if an error is related to network connectivity."""
        network_error_keywords = [
            "connection reset", "connection error", "timeout",
            "network", "[errno", "socket", "ssl"
        ]
        return any(keyword in error_msg.lower() for keyword in network_error_keywords)

    @staticmethod
    def is_rate_limit_error(error_msg: str) -> bool:
        """Check if an error is related to rate limiting."""
        rate_limit_keywords = [
            "rate limit", "429", "too many requests", "quota",
            "resource_exhausted", "ratelimit", "exceeded your current quota"
        ]
        return any(keyword in error_msg.lower() for keyword in rate_limit_keywords)

    @staticmethod
    def is_api_error(error_msg: str) -> bool:
        """Check if an error is a recoverable API error."""
        api_error_keywords = [
            "list index out of range", "invalid response", "empty response",
            "invalid format", "outputparserexception", "crewai.agents.parser"
        ]
        return any(keyword in error_msg.lower() for keyword in api_error_keywords)

    @staticmethod
    def is_retryable_error(error_msg: str) -> bool:
        """Check if an error is retryable."""
        return (ErrorClassifier.is_network_error(error_msg) or 
                ErrorClassifier.is_rate_limit_error(error_msg) or 
                ErrorClassifier.is_api_error(error_msg))


class RetryHandler:
    """Handle retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 5):
        """Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
    
    def handle_network_error(self, error: Exception, retry_count: int) -> bool:
        """Handle network connectivity errors with exponential backoff.
        
        Args:
            error: The exception that occurred
            retry_count: Current retry attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        if retry_count <= self.max_retries:
            # Calculate wait time with exponential backoff and jitter
            base_wait_time = min(2 ** (retry_count + 2), 300)  # Cap at 5 minutes
            jitter = random.uniform(0.8, 1.2)  # Add ±20% jitter
            wait_time = base_wait_time * jitter

            logger.error(f"Network connectivity error: {error}")
            logger.info(f"Global retry {retry_count}/{self.max_retries}. Waiting {wait_time:.1f} seconds before retry...")
            time.sleep(wait_time)
            return True
        return False

    def handle_rate_limit_error(self, error: Exception, retry_count: int) -> bool:
        """Handle rate limiting errors with longer wait times.
        
        Args:
            error: The exception that occurred
            retry_count: Current retry attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        if retry_count <= self.max_retries:
            # For rate limits, wait much longer - start at 60 seconds
            base_wait_time = min(60 * (2 ** retry_count), 600)  # Cap at 10 minutes
            jitter = random.uniform(0.9, 1.1)  # Smaller jitter for rate limits
            wait_time = base_wait_time * jitter

            logger.error(f"Rate limit error: {error}")
            logger.info(f"Rate limit retry {retry_count}/{self.max_retries}. Waiting {wait_time:.1f} seconds before retry...")
            time.sleep(wait_time)
            return True
        return False

    def handle_api_error(self, error: Exception, retry_count: int) -> bool:
        """Handle API errors with moderate wait times.
        
        Args:
            error: The exception that occurred
            retry_count: Current retry attempt number
            
        Returns:
            True if should retry, False otherwise
        """
        if retry_count <= self.max_retries:
            # For API errors, use moderate wait times
            wait_time = min(10 * retry_count, 60)  # 10s, 20s, 30s, up to 60s
            logger.info(f"API error retry {retry_count}/{self.max_retries}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return True
        return False


def with_retry(max_retries: int = 5, 
               handle_network: bool = True,
               handle_rate_limit: bool = True,
               handle_api: bool = True):
    """Decorator to add retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        handle_network: Whether to handle network errors
        handle_rate_limit: Whether to handle rate limit errors
        handle_api: Whether to handle API errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_handler = RetryHandler(max_retries)
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    logger.info(f"Executing {func.__name__} (attempt {retry_count + 1}/{max_retries + 1})")
                    result = func(*args, **kwargs)
                    if retry_count > 0:
                        logger.info(f"{func.__name__} succeeded after {retry_count} retries")
                    return result
                    
                except Exception as e:
                    error_msg = str(e)
                    retry_count += 1
                    
                    logger.error(f"Error in {func.__name__} on attempt {retry_count}: {e}")
                    
                    # Check if this is a rate limit error (highest priority for longer waits)
                    if handle_rate_limit and ErrorClassifier.is_rate_limit_error(error_msg):
                        if retry_handler.handle_rate_limit_error(e, retry_count):
                            continue
                        logger.error("Rate limit errors persist after all retries")
                        raise RateLimitError(f"Rate limit errors persist after {max_retries} retries. Please wait longer and try again.") from e
                    
                    # Check if this is a network connectivity error
                    elif handle_network and ErrorClassifier.is_network_error(error_msg):
                        if retry_handler.handle_network_error(e, retry_count):
                            continue
                        logger.error("Network errors persist after all retries")
                        raise NetworkError(f"Network connectivity issues persist after {max_retries} retries.") from e
                    
                    # Check if this is a recoverable API error
                    elif handle_api and ErrorClassifier.is_api_error(error_msg):
                        if retry_handler.handle_api_error(e, retry_count):
                            continue
                        logger.error("API errors persist after all retries")
                        raise APIError(f"API parsing errors persist after {max_retries} retries. This may indicate a system issue.") from e
                    
                    # For other errors, fail immediately unless it's the last attempt
                    else:
                        if retry_count > max_retries:
                            raise SalesGuruError(f"Unrecoverable error after {max_retries} attempts: {e}") from e
                        else:
                            # Give other errors one more chance with a short wait
                            logger.warning(f"Unknown error type, waiting 5 seconds before retry: {e}")
                            time.sleep(5)
                            continue
            
            # If we get here, we've exhausted all retries
            raise SalesGuruError(f"Failed to execute {func.__name__} after {max_retries} retries due to persistent errors.")
        
        return wrapper
    return decorator


class ErrorReporter:
    """Report and log errors in a structured way."""
    
    @staticmethod
    def log_error(error: Exception, context: str = "", additional_info: dict = None) -> None:
        """Log an error with context and additional information.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            additional_info: Additional information about the error
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        if additional_info:
            error_info.update(additional_info)
        
        logger.error(f"Error occurred: {error_info}")
    
    @staticmethod
    def log_recovery(action: str, context: str = "") -> None:
        """Log successful error recovery.
        
        Args:
            action: The recovery action taken
            context: Context where recovery occurred
        """
        logger.info(f"Error recovery successful: {action} in {context}")


def handle_exceptions(func: Callable) -> Callable:
    """Decorator to handle exceptions and provide structured error reporting.
    
    Args:
        func: Function to wrap with exception handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except (RateLimitError, NetworkError, APIError, ConfigurationError) as e:
            # These are our custom exceptions, re-raise them
            ErrorReporter.log_error(e, func.__name__)
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            ErrorReporter.log_error(e, func.__name__)
            raise SalesGuruError(f"Unexpected error in {func.__name__}: {e}") from e
    
    return wrapper 