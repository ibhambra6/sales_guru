"""Sales Guru crew implementation with task completion guarantees."""

import os
import logging
import json
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, CSVSearchTool, FileReadTool, ScrapeWebsiteTool

from sales_guru.config import config
from sales_guru.llm_manager import llm_manager
from sales_guru.task_monitor import TaskCompletionMonitor, ensure_task_completion
from sales_guru.tools import TaskValidatorTool, JSONToMarkdownTool, ExampleGeneratorTool
from sales_guru.schemas import (
    LeadQualificationResponse, ProspectResearchResponse,
    EmailOutreachResponse, SalesCallPrepResponse
)
from sales_guru.error_handling import handle_exceptions, with_retry

# Configure logging
logger = logging.getLogger("SalesGuru")

# Initialize tools (will be updated with dynamic CSV path in __init__)
web_search_tool = SerperDevTool()
task_validator_tool = TaskValidatorTool()
json_to_markdown_tool = JSONToMarkdownTool()
scrape_web_tool = ScrapeWebsiteTool()
example_generator_tool = ExampleGeneratorTool()

# CSV tools will be initialized dynamically based on input file
csv_search_tool = None
csv_read_tool = None


@CrewBase
class SalesGuru:
    """SalesGuru crew with task completion guarantees."""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, csv_file_path: str = None):
        """Initialize the SalesGuru crew with task completion monitoring and dynamic CSV file path.
        
        Args:
            csv_file_path: Path to the CSV file containing leads data
        """
        # Set default CSV file path
        self.csv_file_path = csv_file_path or config.default_csv_path

        # Initialize CSV tools with the specified file path
        self._initialize_csv_tools()

        # Initialize task monitoring with conservative settings to avoid rate limits
        self.task_monitor = TaskCompletionMonitor(
            max_retries=2,  # Reduced retries to avoid hitting rate limits
            enable_logging=True,
            model="gpt-4.1",  # Use more stable model for monitoring
            token_limit_per_min=15000  # More conservative token limit
        )

    def _initialize_csv_tools(self) -> None:
        """Initialize CSV tools with the current file path."""
        global csv_search_tool, csv_read_tool
        csv_search_tool = CSVSearchTool(csv=self.csv_file_path)
        csv_read_tool = FileReadTool(file_path=self.csv_file_path)
        logger.info(f"CSV tools initialized with file: {self.csv_file_path}")

    def _update_csv_file_path(self, new_path: str) -> None:
        """Update the CSV file path and reinitialize tools.
        
        Args:
            new_path: New path to the CSV file
        """
        if new_path != self.csv_file_path:
            logger.info(f"Updating CSV file path from {self.csv_file_path} to {new_path}")
            self.csv_file_path = new_path
            self._initialize_csv_tools()

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def lead_qualification(self) -> Agent:
        """Create lead qualification agent."""
        agent = Agent(
            config=config.agents_config['lead_qualification'],
            tools=[csv_search_tool, csv_read_tool, web_search_tool, task_validator_tool],
            verbose=True,
            llm=llm_manager.get_lead_qualification_llm(),
            allow_delegation=False,
            response_format=LeadQualificationResponse
        )
        # Enhance the agent with completion guarantees
        return self.task_monitor.enhance_agent(agent)

    @agent
    def prospect_research(self) -> Agent:
        """Create prospect research agent."""
        agent = Agent(
            config=config.agents_config['prospect_research'],
            tools=[web_search_tool, task_validator_tool, scrape_web_tool],
            verbose=True,
            llm=llm_manager.get_google_llm(),
            allow_delegation=False,
            response_format=ProspectResearchResponse
        )
        # Enhance the agent with completion guarantees
        return self.task_monitor.enhance_agent(agent)

    @agent
    def email_outreach(self) -> Agent:
        """Create email outreach agent."""
        agent = Agent(
            config=config.agents_config['email_outreach'],
            tools=[task_validator_tool, example_generator_tool],
            verbose=True,
            llm=llm_manager.get_openai_llm(),
            allow_delegation=False,
            response_format=EmailOutreachResponse
        )
        # Enhance the agent with completion guarantees
        return self.task_monitor.enhance_agent(agent)

    @agent
    def sales_call_prep(self) -> Agent:
        """Create sales call preparation agent."""
        agent = Agent(
            config=config.agents_config['sales_call_prep'],
            tools=[task_validator_tool, example_generator_tool],
            verbose=True,
            llm=llm_manager.get_openai_llm(),
            allow_delegation=False,
            response_format=SalesCallPrepResponse
        )
        # Enhance the agent with completion guarantees
        return self.task_monitor.enhance_agent(agent)

    @agent
    def supervisor_agent(self) -> Agent:
        """Create supervisor agent."""
        agent = Agent(
            config=config.agents_config['supervisor'],
            verbose=True,
            llm=llm_manager.get_google_llm(),
            allow_delegation=True
        )
        return self.task_monitor.enhance_agent(agent)

    @agent
    def markdown_conversion(self) -> Agent:
        """Create markdown conversion agent."""
        return self.task_monitor.enhance_agent(Agent(
            config=config.agents_config['markdown_conversion_agent'],
            tools=[json_to_markdown_tool],
            verbose=True,
            llm=llm_manager.get_google_llm(),
            allow_delegation=False
        ))

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def lead_qualification_task(self) -> Task:
        """Create lead qualification task."""
        task = Task(
            config=config.tasks_config['lead_qualification_task'],
            agent=self.lead_qualification(),
            output_json=LeadQualificationResponse
        )
        # Enhance the task with completion guarantees
        return self.task_monitor.enhance_task(task)

    @task
    def markdown_lead_qualification_task(self) -> Task:
        """Create markdown lead qualification task."""
        return self.task_monitor.enhance_task(Task(
            config=config.tasks_config['markdown_lead_qualification_task'],
            agent=self.markdown_conversion(),
            context=[self.lead_qualification_task()],
            output_file=config.tasks_config['markdown_lead_qualification_task']['output_file']
        ))

    @task
    def prospect_research_task(self) -> Task:
        """Create prospect research task."""
        task = Task(
            config=config.tasks_config['prospect_research_task'],
            agent=self.prospect_research(),
            context=[self.lead_qualification_task()],
            output_json=ProspectResearchResponse
        )
        # Enhance the task with completion guarantees
        return self.task_monitor.enhance_task(task)

    @task
    def markdown_prospect_research_task(self) -> Task:
        """Create markdown prospect research task."""
        return self.task_monitor.enhance_task(Task(
            config=config.tasks_config['markdown_prospect_research_task'],
            agent=self.markdown_conversion(),
            context=[self.prospect_research_task()],
            output_file=config.tasks_config['markdown_prospect_research_task']['output_file']
        ))

    @task
    def email_outreach_task(self) -> Task:
        """Create email outreach task."""
        task = Task(
            config=config.tasks_config['email_outreach_task'],
            agent=self.email_outreach(),
            context=[self.prospect_research_task(), self.lead_qualification_task()],
            output_json=EmailOutreachResponse
        )
        # Enhance the task with completion guarantees
        return self.task_monitor.enhance_task(task)

    @task
    def markdown_email_outreach_task(self) -> Task:
        """Create markdown email outreach task."""
        return self.task_monitor.enhance_task(Task(
            config=config.tasks_config['markdown_email_outreach_task'],
            agent=self.markdown_conversion(),
            context=[self.email_outreach_task()],
            output_file=config.tasks_config['markdown_email_outreach_task']['output_file']
        ))

    @task
    def sales_call_prep_task(self) -> Task:
        """Create sales call preparation task."""
        task = Task(
            config=config.tasks_config['sales_call_prep_task'],
            agent=self.sales_call_prep(),
            context=[self.email_outreach_task(), self.prospect_research_task(), self.lead_qualification_task()],
            output_json=SalesCallPrepResponse
        )
        # Enhance the task with completion guarantees
        return self.task_monitor.enhance_task(task)

    @task
    def markdown_sales_call_prep_task(self) -> Task:
        """Create markdown sales call preparation task."""
        return self.task_monitor.enhance_task(Task(
            config=config.tasks_config['markdown_sales_call_prep_task'],
            agent=self.markdown_conversion(),
            context=[self.sales_call_prep_task()],
            output_file=config.tasks_config['markdown_sales_call_prep_task']['output_file']
        ))

    @crew
    def crew(self) -> Crew:
        """Creates the SalesGuru crew with task completion guarantees."""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[
                self.lead_qualification(), self.prospect_research(),
                self.email_outreach(), self.sales_call_prep(),
                self.markdown_conversion()
            ],
            tasks=[
                self.lead_qualification_task(), self.markdown_lead_qualification_task(),
                self.prospect_research_task(), self.markdown_prospect_research_task(),
                self.email_outreach_task(), self.markdown_email_outreach_task(),
                self.sales_call_prep_task(), self.markdown_sales_call_prep_task()
            ],
            manager_llm=llm_manager.get_google_llm(),
            manager_agent=self.supervisor_agent(),
            verbose=True,
            process=Process.hierarchical,  # Using hierarchical process for manager supervision
        )

    @ensure_task_completion
    @with_retry(max_retries=5)
    @handle_exceptions
    def kickoff(self, inputs: dict = None) -> dict:
        """Run the crew with task completion guarantees.
        
        Args:
            inputs: Input parameters for the crew execution
            
        Returns:
            Results from the crew execution
        """
        # Extract file_name from inputs if provided and update CSV tools
        if inputs and 'file_name' in inputs:
            self._update_csv_file_path(inputs['file_name'])

        # The @ensure_task_completion decorator will make sure all tasks are completed
        result = self.crew().kickoff(inputs=inputs)
        
        # Ensure we never return None, which would cause 'NoneType' object is not callable errors
        if result is None:
            logger.info("CrewAI kickoff returned None, replacing with empty dict")
            return {}
        return result

    @ensure_task_completion
    @with_retry(max_retries=3)
    @handle_exceptions
    def train(self, n_iterations: int = 1, filename: str = None, inputs: dict = None) -> dict:
        """Train the crew with task completion guarantees.
        
        Args:
            n_iterations: Number of training iterations
            filename: Optional filename to save training results
            inputs: Input parameters for training
            
        Returns:
            Training results
        """
        # Note: CrewAI's train method might not be fully compatible with hierarchical process and manager_agent.
        # This is a placeholder and might need adjustments based on CrewAI's capabilities.
        if hasattr(self.crew(), 'train'):
            result = self.crew().train(n_iterations=n_iterations, inputs=inputs)
            if result is None:
                logger.info("CrewAI train returned None, replacing with empty dict")
                return {}
            
            # If a filename is provided, attempt to save the result (e.g., as JSON)
            if filename and result:
                try:
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"Training results saved to {filename}")
                except Exception as e:
                    logger.error(f"Error saving training results to {filename}: {e}")
            return result
        else:
            logger.warning("The 'train' method is not available for this crew configuration.")
            return {}

    @ensure_task_completion
    @with_retry(max_retries=3)
    @handle_exceptions
    def replay(self, task_id: str = None) -> dict:
        """Replay a specific task with task completion guarantees.
        
        Args:
            task_id: ID of the task to replay
            
        Returns:
            Replay results
        """
        # Note: Replay might have limitations with hierarchical processes.
        if hasattr(self.crew(), 'replay_from_task_id'):
            result = self.crew().replay_from_task_id(task_id=task_id)
            if result is None:
                logger.info("CrewAI replay returned None, replacing with empty dict")
                return {}
            return result
        else:
            logger.warning("The 'replay_from_task_id' method is not available for this crew configuration.")
            return {}

    @ensure_task_completion
    @with_retry(max_retries=3)
    @handle_exceptions
    def test(self, n_iterations: int = 1, openai_model_name: str = None, inputs: dict = None) -> dict:
        """Test the crew with task completion guarantees.
        
        Args:
            n_iterations: Number of test iterations
            openai_model_name: OpenAI model name to use for testing
            inputs: Input parameters for testing
            
        Returns:
            Test results
        """
        # Note: Test method might not be fully compatible with hierarchical process.
        # This is a placeholder.
        logger.warning("The 'test' method is not fully implemented for hierarchical crews. Running kickoff instead.")
        return self.kickoff(inputs=inputs)
