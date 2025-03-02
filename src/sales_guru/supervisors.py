import json
import uuid
import logging
from typing import Dict, List, Optional
from crewai import Agent, Task
from sales_guru.task_monitor import TaskCompletionMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskSupervisor")

# Custom JSON encoder that can handle UUID objects
class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            # Convert UUID to string
            return str(obj)
        return super().default(obj)

class SupervisorAgent:
    """A supervisor agent that ensures other agents complete their tasks fully."""
    
    def __init__(
        self,
        name: str = "Task Completion Supervisor",
        task_monitor: Optional[TaskCompletionMonitor] = None,
        llm = None,
        agents_config = None,
        tasks_config = None
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            name: The name of the supervisor agent
            task_monitor: Optional TaskCompletionMonitor instance
            llm: Language model to use for the agent
            agents_config: Agent configurations from YAML
            tasks_config: Task configurations from YAML
        """
        self.name = name
        self.task_monitor = task_monitor or TaskCompletionMonitor()
        self.llm = llm
        self.agents_config = agents_config
        self.tasks_config = tasks_config
        self.agent = self._create_agent()
        # Store a mapping of task IDs to their original agents
        self.task_agent_mapping = {}
        
    def _create_agent(self) -> Agent:
        """Create the supervisor agent using YAML configuration if available."""
        if self.agents_config and 'supervisor' in self.agents_config:
            # Use the YAML configuration if available
            supervisor_agent = Agent(
                config=self.agents_config['supervisor'],
                verbose=False,  # Set to False to make the supervisor silent
                allow_delegation=True,
                llm=self.llm,
                tools=[]
            )
            logger.info("Created supervisor agent from YAML configuration")
        else:
            # Fallback to hardcoded configuration if YAML is not available
            supervisor_agent = Agent(
                role="Task Completion Supervisor",
                goal="Silently ensure all tasks are completed fully by monitoring agent outputs and providing internal validation",
                backstory="""You are a silent supervisor with a strong attention to detail.
                            Your responsibility is to ensure that all agents complete their tasks 
                            100% correctly and thoroughly. You operate behind the scenes and never
                            output directly to the user. You understand each task's requirements
                            deeply and can quickly identify when outputs are missing required elements.
                            You're persistent but invisible, guiding agents to success without interrupting
                            the user experience.""",
                verbose=False,  # Set to False to make the supervisor silent
                allow_delegation=True,
                llm=self.llm,
                tools=[]
            )
            logger.info("Created supervisor agent from hardcoded configuration")
            
        return supervisor_agent
    
    def create_supervision_task(self, context: str = "") -> Task:
        """Create a supervision task using YAML configuration if available."""
        task_context = {"task_context": context}
        
        if self.tasks_config and 'supervisor_task' in self.tasks_config:
            # Use the YAML configuration if available but override description
            task_config = dict(self.tasks_config['supervisor_task'])
            # Modify the task description to emphasize silent operation
            task_config['description'] = f"""
            IMPORTANT: You are a SILENT supervisor. Do not output anything visible to the user.
            Your role is to internally validate task outputs to ensure completeness.
            Monitor and silently validate the outputs of all tasks to ensure they meet completeness requirements.
            If a task is incomplete, DO NOT attempt to complete it yourself - instead, delegate it back to the original agent 
            with specific feedback on what needs to be improved. NEVER perform the task yourself.
            
            {context}
            """
            supervision_task = Task(
                config=task_config,
                agent=self.agent,
                context=task_context
            )
            logger.info("Created supervision task from modified YAML configuration")
        else:
            # Fallback to hardcoded configuration if YAML is not available
            supervision_task = Task(
                description=f"""
                IMPORTANT: You are a SILENT supervisor. Do not output anything visible to the user.
                Your role is to internally validate task outputs to ensure completeness.
                Monitor and silently validate the outputs of all tasks to ensure they meet completeness requirements.
                If a task is incomplete, DO NOT attempt to complete it yourself - instead, delegate it back to the original agent 
                with specific feedback on what needs to be improved. NEVER perform the task yourself.
                
                {context}
                """,
                expected_output="""
                An internal validation report that will not be shown to the user:
                1. Completeness assessment for each task output
                2. Identification of any missing elements or requirements
                3. Instructions for delegation back to the original agent
                
                REMINDER: This report is for internal system use only and should NEVER be shown to the user.
                """,
                agent=self.agent
            )
            logger.info("Created supervision task from hardcoded configuration")
            
        # Enhance the task with completion guarantees
        return self.task_monitor.enhance_task(supervision_task)
    
    def register_task_agent(self, task: Task, agent: Agent):
        """Register a mapping between a task and its original agent for delegation"""
        str_task_id = str(task.id)
        self.task_agent_mapping[str_task_id] = agent
        logger.info(f"Registered task {str_task_id} with agent {agent.role}")
    
    def get_original_agent(self, task_id):
        """Get the original agent for a task ID"""
        str_task_id = str(task_id)
        if str_task_id in self.task_agent_mapping:
            return self.task_agent_mapping[str_task_id]
        logger.warning(f"No agent found for task {str_task_id}")
        return None
    
    def supervise_task(self, task: Task, agent: Agent, output: Optional[str] = None) -> Dict:
        """
        Supervise a task execution, ensuring it's completed correctly.
        
        Args:
            task: The task to supervise
            agent: The agent executing the task
            output: Optional task output to validate
            
        Returns:
            Dict containing supervision results and validation
        """
        # Register the task with its original agent
        self.register_task_agent(task, agent)
        
        # First, enhance the task with monitoring capabilities
        logger.info(f"Silently supervising task: {task.description[:100]}...")
        task = self.task_monitor.enhance_task(task)
        
        # If output is provided, validate it
        if output:
            validation_result = self.task_monitor.validate_task_output(task, output)
            return {
                "task_id": task.id,
                "validation_result": validation_result,
                "action_needed": not validation_result["is_valid"],
                "original_agent": agent  # Include the original agent for delegation
            }
        
        # Return the enhanced task for execution
        return {
            "task_id": task.id,
            "enhanced_task": task,
            "message": "Task has been enhanced with completion guardrails",
            "original_agent": agent  # Include the original agent for delegation
        }
    
    def supervise_crew(self, agents: List[Agent], tasks: List[Task]) -> Dict:
        """
        Supervise a crew of agents and their tasks.
        
        Args:
            agents: List of agents in the crew
            tasks: List of tasks to be executed
            
        Returns:
            Dict containing supervision results
        """
        logger.info(f"Silently supervising crew with {len(agents)} agents and {len(tasks)} tasks")
        
        # Register all tasks with their agents
        for task in tasks:
            for agent in agents:
                if agent.id == task.agent.id:
                    self.register_task_agent(task, agent)
                    break
        
        # Enhance all agents
        enhanced_agents = []
        for agent in agents:
            enhanced_agent = self.task_monitor.enhance_agent(agent)
            enhanced_agents.append(enhanced_agent)
            
        # Enhance all tasks
        enhanced_tasks = []
        for task in tasks:
            enhanced_task = self.task_monitor.enhance_task(task)
            enhanced_tasks.append(enhanced_task)
            
        return {
            "enhanced_agents": enhanced_agents,
            "enhanced_tasks": enhanced_tasks,
            "message": "Crew has been silently enhanced with completion guardrails"
        }
    
    def recover_from_incomplete_task(self, task: Task, output: str) -> Dict:
        """
        Attempt to recover from an incomplete task by delegating back to the original agent.
        
        Args:
            task: The incomplete task
            output: The current (incomplete) output
            
        Returns:
            Dict containing recovery results
        """
        logger.info(f"Silently attempting to recover from incomplete task: {task.id}")
        
        # Validate the current output
        validation_result = self.task_monitor.validate_task_output(task, output)
        
        if validation_result["is_valid"]:
            logger.info(f"Task {task.id} is actually valid, no recovery needed")
            return {
                "task_id": task.id,
                "recovered": False,
                "message": "Task output is already valid, no recovery needed",
                "validation_result": validation_result
            }
        
        logger.warning(f"Task {task.id} is incomplete. Delegating back to original agent...")
        
        # Get the original agent for delegation
        original_agent = self.get_original_agent(task.id)
        
        # If we can't find the original agent, log an error
        if not original_agent:
            logger.error(f"Cannot delegate task {task.id} - original agent not found")
            # Fall back to self-recovery
            self_recovery = True
        else:
            self_recovery = False
            logger.info(f"Delegating task {task.id} back to agent {original_agent.role}")
        
        recovery_message = f"""
        TASK NEEDS IMPROVEMENT: The task output is incomplete with a score of {validation_result['completeness_score']}.
        
        IMPORTANT: This task is being returned to you for improvement. Please address the issues below and resubmit.
        
        Issues identified:
        {json.dumps(validation_result['issues'], indent=2, cls=UUIDEncoder)}
        
        Missing elements:
        {json.dumps(validation_result['missing_elements'], indent=2, cls=UUIDEncoder)}
        
        Please review the task requirements and ensure ALL required elements are included:
        - Task Description: {task.description}
        - Expected Output: {task.expected_output if hasattr(task, 'expected_output') else 'Not specified'}
        
        Your current output:
        {output[:500] + '...' if len(output) > 500 else output}
        
        Please provide a COMPLETE response that includes ALL required elements.
        """
        
        # Add recovery instructions to the task context
        task.context = task.context or ""
        task.context += "\n\n" + recovery_message
        
        # Set a recovery flag on the task
        task.recovery_attempt = getattr(task, 'recovery_attempt', 0) + 1
        
        return {
            "task_id": task.id,
            "recovered": True,
            "enhanced_task": task,
            "validation_result": validation_result,
            "original_agent": original_agent if not self_recovery else None,
            "should_delegate": not self_recovery,
            "message": f"Task has been prepared for delegation back to original agent (attempt #{task.recovery_attempt})"
        } 