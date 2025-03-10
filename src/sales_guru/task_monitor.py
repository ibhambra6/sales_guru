import json
import logging
import uuid
import time
import tiktoken
from functools import wraps
from typing import Callable
from crewai import Task, Agent
from crewai.task import TaskOutput
import importlib

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

class TaskCompletionMonitor:
    """
    A simplified monitor to ensure tasks are completed successfully.
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
        Create a simple callback function to monitor task execution.
        
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
            if hasattr(output, 'result'):
                result_value = output.result
            elif hasattr(output, 'output'):
                result_value = output.output
            elif hasattr(output, 'raw_output'):
                result_value = output.raw_output
            else:
                result_value = output
            
            # Log the attempt
            if self.enable_logging:
                logger.info(f"Task {task_id} - Attempt {attempt_number} completed")
            
            # Store execution history
            self.task_history[task_id].append({
                "attempt": attempt_number,
                "timestamp": output.created_at.isoformat() if hasattr(output, "created_at") else None
            })
            
            # Simple validation: just check if output exists and isn't empty
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
    
    def enhance_task(self, task: Task) -> Task:
        """
        Enhance a task with simple monitoring and completion guarantees.
        
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
        
        # Add simple completion instructions
        if hasattr(task, "description") and task.description:
            task.description = task.description.strip()
            task.description += "\n\nIMPORTANT: You MUST complete this task fully with all required information."
        
        return task
    
    def enhance_agent(self, agent: Agent) -> Agent:
        """
        Add simple enhancements to an agent.
        
        Args:
            agent: The CrewAI agent to enhance
            
        Returns:
            The enhanced agent
        """
        # Add basic completion reminder to agent's goal
        if agent.goal:
            agent.goal = agent.goal.strip()
            agent.goal += " Complete all tasks thoroughly and accurately."
        
        return agent
    
    def log_task_summary(self):
        """Generate a simple log summary of task execution status"""
        if not self.enable_logging:
            return
            
        total_tasks = len(self.task_history)
        tasks_with_retries = sum(1 for task_id, attempts in self.task_attempt_counter.items() if attempts > 1)
        
        logger.info(f"Task Execution Summary: {total_tasks} tasks completed")
        logger.info(f"Tasks requiring retries: {tasks_with_retries}")
        
        for task_id, attempts in self.task_attempt_counter.items():
            logger.info(f"Task {task_id}: {attempts} attempt(s)")


def ensure_task_completion(func):
    """
    Simplified decorator to handle common task completion errors.
    Focuses on empty responses and rate limits.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the monitor instance from the class if available
        self_arg = args[0] if args else None
        monitor = getattr(self_arg, "task_monitor", None)
        
        if not monitor:
            # Create a new monitor if one doesn't exist
            monitor = TaskCompletionMonitor()
        
        try:
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Log task summary
            monitor.log_task_summary()
            
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limit errors
            if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                logger.warning(f"Rate limit error encountered: {e}")
                
                # Wait for a reset
                wait_time = 60
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Try again
                return func(*args, **kwargs)
                
            # Handle empty response errors
            elif "invalid response" in error_msg and "empty" in error_msg:
                logger.warning(f"Empty response error encountered: {e}")
                
                # Wait some time before retrying
                wait_time = 30
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Try again
                return func(*args, **kwargs)
                
            else:
                # Re-raise other errors
                raise
    
    return wrapper


class TokenMonitor:
    """Simplified utility to monitor token usage and prevent rate limit errors"""
    
    def __init__(self, model="gpt-4o", token_limit_per_min=28000):
        """
        Initialize the token monitor.
        
        Args:
            model: The model name to use for token counting
            token_limit_per_min: Token limit per minute
        """
        self.model = model
        self.token_limit_per_min = token_limit_per_min
        self.tokens_used_in_window = 0
        self.window_start_time = time.time()
        self.encoding = tiktoken.encoding_for_model(model) if model.startswith("gpt") else tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
        
    def wait_for_token_reset(self):
        """Wait until the rate limit window resets"""
        current_time = time.time()
        seconds_since_window_start = current_time - self.window_start_time
        
        if seconds_since_window_start < 60:
            # Calculate how many seconds to wait
            wait_time = 60 - seconds_since_window_start
            logger.warning(f"Approaching rate limit. Waiting {wait_time:.1f} seconds for reset")
            time.sleep(wait_time)
            
        # Reset the window
        self.tokens_used_in_window = 0
        self.window_start_time = time.time()
    
    def truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens"""
        if not text:
            return ""
            
        encoded = self.encoding.encode(text)
        if len(encoded) <= max_tokens:
            return text
            
        # Truncate and add an indicator
        truncated_encoded = encoded[:max_tokens-3]
        truncated_text = self.encoding.decode(truncated_encoded)
        return truncated_text + "..." 