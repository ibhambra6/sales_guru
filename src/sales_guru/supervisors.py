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
        supervisor_config = None
        
        if self.agents_config and 'supervisor' in self.agents_config:
            supervisor_config = self.agents_config['supervisor']
            logger.info("Using supervisor agent from YAML configuration")
        
        supervisor_agent = Agent(
            config=supervisor_config if supervisor_config else None,
            role=supervisor_config.get('role', "Task Completion Supervisor") if supervisor_config else "Task Completion Supervisor",
            goal=supervisor_config.get('goal', "Ensure all tasks are completed fully and accurately") if supervisor_config else "Silently ensure all tasks are completed fully by monitoring agent outputs and providing internal validation",
            backstory=supervisor_config.get('backstory', """You are a silent supervisor with a strong attention to detail.
                      Your responsibility is to ensure that all agents complete their tasks 
                      100% correctly and thoroughly. You operate behind the scenes and never
                      output directly to the user.""") if supervisor_config else """You are a silent supervisor with a strong attention to detail.
                      Your responsibility is to ensure that all agents complete their tasks 
                      100% correctly and thoroughly. You operate behind the scenes and never
                      output directly to the user. You understand each task's requirements
                      deeply and can quickly identify when outputs are missing required elements.
                      You're persistent but invisible, guiding agents to success without interrupting
                      the user experience.""",
            verbose=False,  # Supervisor should be silent
            allow_delegation=True,
            llm=self.llm,
            tools=[]
        )
        
        return supervisor_agent
    
    def create_supervision_task(self, context: str = "") -> Task:
        """Create a supervision task using YAML configuration if available."""
        task_context = {"task_context": context}
        
        # Common description for supervision task
        supervision_description = f"""
        IMPORTANT: You are a SILENT supervisor. Do not output anything visible to the user.
        Your role is to internally validate task outputs to ensure completeness.
        Monitor and silently validate the outputs of all tasks to ensure they meet completeness requirements.
        If a task is incomplete, DO NOT attempt to complete it yourself - instead, delegate it back to the original agent 
        with specific feedback on what needs to be improved. NEVER perform the task yourself.
        
        {context}
        """
        
        if self.tasks_config and 'supervisor_task' in self.tasks_config:
            # Use the YAML configuration but override description
            task_config = dict(self.tasks_config['supervisor_task'])
            task_config['description'] = supervision_description
            supervision_task = Task(
                config=task_config,
                agent=self.agent,
                context=task_context
            )
        else:
            # Fallback to hardcoded configuration
            supervision_task = Task(
                description=supervision_description,
                expected_output="""
                An internal validation report that will not be shown to the user:
                1. Completeness assessment for each task output
                2. Identification of any missing elements or requirements
                3. Instructions for delegation back to the original agent
                
                REMINDER: This report is for internal system use only and should NEVER be shown to the user.
                """,
                agent=self.agent
            )
        
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
        
        # Enhance the task with monitoring capabilities
        logger.info(f"Silently supervising task: {task.description[:100]}...")
        task = self.task_monitor.enhance_task(task)
        
        # If output is provided, validate it
        if output:
            validation_result = self.task_monitor.validate_task_output(task, output)
            return {
                "task_id": task.id,
                "validation_result": validation_result,
                "action_needed": not validation_result["is_valid"],
                "original_agent": agent
            }
        
        # Return the enhanced task for execution
        return {
            "task_id": task.id,
            "enhanced_task": task,
            "message": "Task has been enhanced with completion guardrails",
            "original_agent": agent
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
        for i, task in enumerate(tasks):
            if i < len(agents):
                agent = agents[i]
                self.register_task_agent(task, agent)
        
        # Enhance all tasks with monitoring capabilities
        enhanced_tasks = [self.task_monitor.enhance_task(task) for task in tasks]
        
        # Create a supervision task for the crew
        supervision_task = self.create_supervision_task(
            context=f"Monitor the execution of {len(tasks)} tasks by {len(agents)} agents."
        )
        
        return {
            "enhanced_tasks": enhanced_tasks,
            "supervision_task": supervision_task,
            "message": "Crew has been enhanced with task completion guarantees"
        }
    
    def recover_from_incomplete_task(self, task: Task, output: str) -> Dict:
        """
        Attempt to recover from an incomplete task execution.
        
        Args:
            task: The task that was executed incompletely
            output: The incomplete output
            
        Returns:
            Dict containing recovery results
        """
        logger.warning(f"Attempting to recover from incomplete task: {task.id}")
        
        # Get the original agent for this task
        original_agent = self.get_original_agent(task.id)
        
        if not original_agent:
            logger.error(f"Cannot recover task {task.id}: No agent found")
            return {
                "task_id": task.id,
                "success": False,
                "error": "No agent found for this task"
            }
        
        # Validate the output
        validation_result = self.task_monitor.validate_task_output(task, output)
        
        # If the output is valid, no recovery needed
        if validation_result["is_valid"]:
            return {
                "task_id": task.id,
                "success": True,
                "message": "Output is valid, no recovery needed"
            }
        
        # Add feedback to the task context
        issues = validation_result.get("issues", [])
        recommendations = validation_result.get("recommendations", [])
        
        feedback = "IMPORTANT FEEDBACK FROM SUPERVISOR:\n"
        if issues:
            feedback += f"Issues identified: {', '.join(issues)}\n"
        if recommendations:
            feedback += f"Recommendations: {', '.join(recommendations)}\n"
        
        task.context = (task.context or "") + "\n\n" + feedback
        task.context += "\n\nPlease fix these issues and complete the task fully."
        
        return {
            "task_id": task.id,
            "enhanced_task": task,
            "original_agent": original_agent,
            "validation_result": validation_result,
            "message": "Task has been prepared for recovery"
        } 