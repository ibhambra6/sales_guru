import json
import logging
import uuid
import time
import tiktoken
from functools import wraps
from typing import Callable, Dict, List, Optional, Any, Union
from crewai import Task, Agent
from crewai.task import TaskOutput
import importlib
import random
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskCompletionMonitor")

# Custom JSON encoder that can handle UUID objects
class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            # Convert UUID to string
            return str(obj)
        return super().default(obj)

# Apply monkey patches to the CrewAI library to prevent empty response errors
def apply_crewai_patches():
    """Apply monkey patches to the CrewAI library to prevent empty response errors"""
    try:
        # Import the module that contains the _get_llm_response method
        crew_agent_executor = importlib.import_module("crewai.agents.crew_agent_executor")
        
        # Store the original method
        original_get_llm_response = crew_agent_executor.CrewAgentExecutor._get_llm_response
        
        # Define the patched method
        def patched_get_llm_response(self):
            try:
                # Call the original method
                response = original_get_llm_response(self)
                
                # If response is None or empty, return a fallback
                if response is None or (isinstance(response, str) and not response.strip()):
                    logger.warning("Empty response detected in CrewAgentExecutor._get_llm_response. Using fallback.")
                    return "I apologize, but I couldn't complete this task fully due to technical limitations. Here's a partial response based on my understanding of the request."
                
                return response
            except ValueError as e:
                # If we get a ValueError about empty responses, return a fallback
                if "empty" in str(e).lower():
                    logger.warning(f"Caught ValueError in _get_llm_response: {e}. Using fallback.")
                    return "I apologize, but I encountered an error when processing this task. As a fallback response, I would recommend focusing on the highest priority aspects of this request."
                raise
        
        # Apply the patched method
        crew_agent_executor.CrewAgentExecutor._get_llm_response = patched_get_llm_response
        logger.info("Successfully applied patch to CrewAgentExecutor._get_llm_response")
    except Exception as e:
        logger.error(f"Failed to apply CrewAI patches: {e}")

# Apply the patches when this module is imported
apply_crewai_patches()

class TokenMonitor:
    """Monitor and manage token usage to avoid rate limits"""
    
    def __init__(self, model: str = "gpt-4o", token_limit_per_min: int = 28000):
        """
        Initialize the token monitor.
        
        Args:
            model: The model name to use for encoding/counting tokens
            token_limit_per_min: Maximum tokens per minute to stay under rate limits
        """
        self.model = model
        self.token_limit_per_min = token_limit_per_min
        self.tokens_used_in_current_minute = 0
        self.last_reset_time = time.time()
        
        try:
            self.encoding = tiktoken.encoding_for_model(model.replace("gemini/", ""))
        except Exception:
            # Fallback to cl100k_base encoding if model-specific encoding not available
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
        
    def wait_for_token_reset(self):
        """Wait until token usage is reset based on time elapsed"""
        # Check time since last reset
        elapsed_time = time.time() - self.last_reset_time
        
        # If less than a minute has passed and we're over the limit, wait
        if elapsed_time < 60 and self.tokens_used_in_current_minute >= self.token_limit_per_min:
            wait_time = max(60 - elapsed_time, 0)
            logger.info(f"Rate limit approaching. Waiting {wait_time:.1f} seconds before continuing...")
            time.sleep(wait_time)
            # Reset token count after waiting
            self.tokens_used_in_current_minute = 0
            self.last_reset_time = time.time()
        # If more than a minute has passed, reset the counter
        elif elapsed_time >= 60:
            self.tokens_used_in_current_minute = 0
            self.last_reset_time = time.time()
        
    def truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a maximum token limit"""
        if not text:
            return ""
            
        encoded = self.encoding.encode(text)
        
        if len(encoded) <= max_tokens:
            return text
            
        # Truncate and add a message indicating truncation
        truncated_encoded = encoded[:max_tokens-50]  # Leave room for message
        truncated_text = self.encoding.decode(truncated_encoded)
        
        return truncated_text + "\n\n[Content was truncated to fit token limits]"


class TaskCompletionMonitor:
    """
    A monitor to ensure tasks are completed successfully.
    Handles empty responses and rate limits gracefully.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        enable_logging: bool = True,
        model: str = "gpt-4o",
        token_limit_per_min: int = 28000
    ):
        """
        Initialize the TaskCompletionMonitor.
        
        Args:
            max_retries: Maximum number of retry attempts for a task
            enable_logging: Whether to log task execution information
            model: The model name to use for token counting
            token_limit_per_min: Maximum tokens per minute to stay under rate limits
        """
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        self.task_attempt_counter = {}
        self.task_history = {}
        
        # Initialize token monitoring
        self.token_monitor = TokenMonitor(model=model, token_limit_per_min=token_limit_per_min)

    def create_task_callback(self, task: Task) -> Callable:
        """
        Create a callback function to monitor task execution.
        
        Args:
            task: The CrewAI task to monitor
            
        Returns:
            A callback function for monitoring this task
        """
        def callback(output: TaskOutput) -> bool:
            task_id = str(task.id)
            
            # Initialize task in the counter if this is the first attempt
            if task_id not in self.task_attempt_counter:
                self.task_attempt_counter[task_id] = 0
                self.task_history[task_id] = []
            
            # Increment attempt counter
            self.task_attempt_counter[task_id] += 1
            attempt_number = self.task_attempt_counter[task_id]
            
            # Extract the output value
            result_value = self._extract_output_value(output)
            
            # Log the attempt
            if self.enable_logging:
                logger.info(f"Task {task_id} - Attempt {attempt_number} completed")
            
            # Store execution history
            self.task_history[task_id].append({
                "attempt": attempt_number,
                "timestamp": output.created_at.isoformat() if hasattr(output, "created_at") else None
            })
            
            # Validate if output exists and isn't empty
            if not result_value:
                if attempt_number < self.max_retries:
                    logger.warning(f"Task {task_id} - Empty result. Retry {attempt_number}/{self.max_retries}")
                    
                    # Add feedback for the next attempt
                    task.context = task.context or ""
                    task.context += "\n\nIMPORTANT: Your previous attempt produced no output. Please ensure you provide a complete response."
                    
                    return False  # Retry the task
                else:
                    logger.error(f"Task {task_id} - Max retries reached with empty results. Proceeding anyway.")
            
            return True  # Task completed successfully
        
        return callback
    
    def _extract_output_value(self, output: Any) -> Any:
        """Extract the actual output value from different output formats"""
        if hasattr(output, 'result'):
            return output.result
        elif hasattr(output, 'output'):
            return output.output
        elif hasattr(output, 'raw_output'):
            return output.raw_output
        return output
    
    def enhance_task(self, task: Task) -> Task:
        """
        Enhance a task with monitoring and completion guarantees.
        
        Args:
            task: The CrewAI task to enhance
            
        Returns:
            The enhanced task with monitoring callback
        """
        # Add monitoring callback to the task
        monitor_callback = self.create_task_callback(task)
        
        # Store original callback if it exists
        original_callback = task.callback if callable(task.callback) else None
        
        # Create a new callback that calls both callbacks
        if original_callback:
            def combined_callback(output):
                original_result = original_callback(output)
                monitor_result = monitor_callback(output)
                return monitor_result and original_result
            task.callback = combined_callback
        else:
            task.callback = monitor_callback
        
        # Add completion instructions
        if hasattr(task, "description") and task.description:
            task.description = task.description.strip()
            task.description += "\n\nIMPORTANT: You MUST complete this task fully with all required information."
        
        return task
    
    def enhance_agent(self, agent: Agent) -> Agent:
        """
        Add enhancements to an agent.
        
        Args:
            agent: The CrewAI agent to enhance
            
        Returns:
            The enhanced agent
        """
        # Add basic completion reminder to agent's goal
        if agent.goal:
            agent.goal = agent.goal.strip()
            if "complete" not in agent.goal.lower() and "thorough" not in agent.goal.lower():
                agent.goal += "\n\nAlways provide complete and thorough responses."
        
        # Add rate limit awareness to agent's backstory
        if agent.backstory:
            agent.backstory = agent.backstory.strip()
            if "rate limit" not in agent.backstory.lower():
                agent.backstory += "\n\nYou are aware of rate limits and will pace your API calls appropriately."
        
        return agent
    
    def validate_task_output(self, task: Task, output: Union[str, Dict, List, BaseModel]) -> Dict:
        """
        Validate if a task output meets requirements.
        
        Args:
            task: The task to validate output for
            output: The output to validate
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            "is_valid": False,
            "issues": [],
            "recommendations": []
        }
        
        # For Pydantic models, validation happens automatically during parsing
        # If we received a Pydantic model, it means it passed validation
        if hasattr(output, '__pydantic_fields__') or hasattr(output, '__pydantic_model__'):
            validation_result["is_valid"] = True
            validation_result["info"] = "Pydantic schema validation passed"
            return validation_result
        
        # Basic validation: check if output exists and isn't empty
        if not output:
            validation_result["issues"].append("Empty or null output")
            validation_result["recommendations"].append("Ensure output contains all required information")
            return validation_result
            
        # Check if the output has a reasonable length
        if isinstance(output, str) and len(output) < 10:
            validation_result["issues"].append("Output is suspiciously short")
            validation_result["recommendations"].append("Provide a more detailed response")
            return validation_result
            
        # If we get here without issues, consider it valid
        validation_result["is_valid"] = True
        return validation_result
    
    def log_task_summary(self):
        """Log a summary of all task attempts"""
        if not self.enable_logging:
            return
            
        for task_id, attempts in self.task_attempt_counter.items():
            logger.info(f"Task {task_id} - Total attempts: {attempts}")
            
        logger.info(f"Total tasks monitored: {len(self.task_attempt_counter)}")


def ensure_task_completion(func):
    """
    Decorator to ensure tasks are completed successfully.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the monitor instance from the class if available
        monitor = None
        if args and hasattr(args[0], 'task_monitor'):
            monitor = args[0].task_monitor
        else:
            # Create a temporary monitor if none exists
            monitor = TaskCompletionMonitor(enable_logging=True)
            
        logger.info(f"Starting execution with task completion guarantees: {func.__name__}")
        
        # Global retry logic
        max_retries = 3
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # Call the original function
                result = func(*args, **kwargs)
                
                # Check if result exists (allowing for falsy results that are not None)
                if result is not None:
                    # Log summary if available
                    if monitor:
                        monitor.log_task_summary()
                    return result
                else:
                    # Empty result, retry
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Empty result after {max_retries} retries")
                        break
                        
                    wait_time = 5 * retries
                    logger.warning(f"Empty result. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            except Exception as e:
                last_error = e
                retries += 1
                error_msg = str(e).lower()
                
                # Check if this is a rate limit or network error
                if any(x in error_msg for x in [
                    "rate limit", "429", "too many requests", 
                    "connection reset", "connection error", "timeout",
                    "network", "empty response"
                ]):
                    if retries <= max_retries:
                        wait_time = min(2 ** (retries + 2), 300)
                        jitter = random.uniform(0.8, 1.2)
                        wait_time = wait_time * jitter
                        
                        logger.warning(f"Recoverable error: {e}. Retry {retries}/{max_retries} in {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                
                # Re-raise the exception for non-recoverable errors
                logger.error(f"Non-recoverable error: {e}")
                raise
        
        # If we get here, we've exhausted all retries
        if last_error:
            raise last_error
        else:
            raise Exception(f"Failed to complete {func.__name__} successfully after {max_retries} retries") 
    
    return wrapper 