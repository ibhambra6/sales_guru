import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, CSVSearchTool, FileReadTool

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
serper_api_key = os.getenv('SERPER_API_KEY')
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Initialize tools
web_search_tool = SerperDevTool()
csv_search_tool = CSVSearchTool(file_path='knowledge/leads.csv')
csv_read_tool = FileReadTool(file_path='knowledge/leads.csv')

@CrewBase
class SalesGuru():
	"""SalesGuru crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def lead_qualification(self) -> Agent:
		return Agent(
			config=self.agents_config['lead_qualification'],
			tools=[csv_search_tool, csv_read_tool, web_search_tool],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def lead_qualification_task(self) -> Task:
		return Task(
			config=self.tasks_config['lead_qualification_task'],
			agent=self.lead_qualification()
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SalesGuru crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
