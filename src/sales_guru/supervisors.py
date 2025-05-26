import json
import uuid
import logging
import os
import csv
from typing import Dict, List, Optional, Any
from crewai import Agent, Task
from sales_guru.task_monitor import TaskCompletionMonitor
from sales_guru.tools import TaskValidatorTool
from sales_guru.schemas import (
    LeadQualificationResponse, ProspectResearchResponse, 
    EmailOutreachResponse, SalesCallPrepResponse
)

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

class SupervisorAgentLogic:
    """Handles the logic for the supervisor agent, including task validation and CSV lead count validation."""
    def __init__(self, llm=None, agents_config=None, tasks_config=None, csv_file_path=None):
        self.task_validator_tool = TaskValidatorTool()
        self.llm = llm
        self.agents_config = agents_config
        self.tasks_config = tasks_config
        self.task_monitor = TaskCompletionMonitor() # Supervisor uses its own monitor instance
        self.csv_file_path = csv_file_path or 'knowledge/leads.csv'  # Default fallback

    def count_csv_leads(self) -> int:
        """Count the number of leads in the CSV file"""
        try:
            with open(self.csv_file_path, 'r') as file:
                reader = csv.reader(file)
                # Skip header row
                next(reader, None)
                # Count remaining rows
                lead_count = sum(1 for _ in reader)
                logger.info(f"Supervisor: Found {lead_count} leads in CSV file: {self.csv_file_path}")
                return lead_count
        except Exception as e:
            logger.error(f"Supervisor: Failed to count leads in CSV {self.csv_file_path}: {e}")
            # Default to a reasonable number if we can't count
            return 25

    def get_csv_lead_data(self) -> List[Dict[str, str]]:
        """Get lead data from the CSV file for validation purposes"""
        csv_leads = []
        try:
            with open(self.csv_file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    csv_leads.append(row)
            logger.info(f"Supervisor: Loaded {len(csv_leads)} lead records from CSV: {self.csv_file_path}")
            return csv_leads
        except Exception as e:
            logger.error(f"Supervisor: Failed to read leads CSV {self.csv_file_path}: {e}")
            return []

    def validate_lead_qualification_output(self, task_output: Any) -> Any:
        """
        Validate lead qualification output specifically to ensure:
        1. Only leads from CSV are included
        2. No fictional leads are created
        3. Lead count matches CSV
        """
        csv_leads = self.get_csv_lead_data()
        expected_count = len(csv_leads)
        
        logger.info(f"Supervisor: Validating lead qualification output. Expected {expected_count} leads from CSV.")
        
        # Handle Pydantic models with qualified_leads attribute
        if hasattr(task_output, 'qualified_leads') and isinstance(task_output.qualified_leads, list):
            actual_count = len(task_output.qualified_leads)
            logger.info(f"Supervisor: Found {actual_count} leads in agent output")
            
            # Validate each lead against CSV data
            valid_leads = []
            invalid_leads = []
            
            for lead in task_output.qualified_leads:
                # Check if this lead's name and company match any in the CSV
                is_valid = any(
                    csv_lead['Name'] == lead.lead_name and 
                    csv_lead['Company Name'] == lead.company_name 
                    for csv_lead in csv_leads
                )
                if is_valid:
                    valid_leads.append(lead)
                else:
                    invalid_leads.append(lead)
                    logger.warning(f"Supervisor: Removing invented lead: {lead.lead_name} from {lead.company_name}")

            # Check for complete failure - agent created only fictional leads
            if len(valid_leads) == 0 and len(invalid_leads) > 0:
                logger.error("Supervisor: CRITICAL ERROR - Agent completely ignored CSV file and created fictional leads!")
                
                expected_leads_str = [f"{csv_lead['Name']} ({csv_lead['Company Name']})" for csv_lead in csv_leads[:5]]
                actual_leads_str = [f"{lead.lead_name} ({lead.company_name})" for lead in invalid_leads[:5]]
                
                logger.error(f"Supervisor: Expected leads from CSV: {expected_leads_str}")
                logger.error(f"Supervisor: Actual leads created: {actual_leads_str}")
                
                # Return error feedback for re-delegation
                return f"SUPERVISOR_FEEDBACK: CRITICAL ERROR - You completely ignored the CSV file and created fictional leads. You must ONLY process the {expected_count} leads from the {self.csv_file_path} file. Expected leads include: {', '.join(expected_leads_str)}. DO NOT CREATE FICTIONAL LEADS."
            
            # Check for missing leads from CSV
            if len(valid_leads) < expected_count:
                missing_leads = []
                for csv_lead in csv_leads:
                    if not any(
                        lead.lead_name == csv_lead['Name'] and 
                        lead.company_name == csv_lead['Company Name'] 
                        for lead in valid_leads
                    ):
                        missing_leads.append(f"{csv_lead['Name']} ({csv_lead['Company Name']})")
                
                logger.warning(f"Supervisor: Missing {len(missing_leads)} leads from CSV: {missing_leads[:5]}")
                
                if missing_leads:
                    return f"SUPERVISOR_FEEDBACK: You are missing {len(missing_leads)} leads from the CSV file. Missing leads: {', '.join(missing_leads[:10])}. You must process ALL {expected_count} leads from the CSV file, not just {len(valid_leads)}."
            
            # Check for excess leads (should not happen if fictional leads are caught)
            if len(valid_leads) > expected_count:
                logger.warning(f"Supervisor: Too many leads in output ({len(valid_leads)} vs expected {expected_count}). Truncating to expected count.")
                task_output.qualified_leads = valid_leads[:expected_count]
            else:
                task_output.qualified_leads = valid_leads
            
            logger.info(f"Supervisor: Lead validation successful. Output contains {len(task_output.qualified_leads)} valid leads.")
            return task_output
                
        # Handle JSON/dict responses
        elif isinstance(task_output, dict) and 'qualified_leads' in task_output:
            # Convert to list if not already
            if not isinstance(task_output['qualified_leads'], list):
                logger.error("Supervisor: qualified_leads should be a list")
                return "SUPERVISOR_FEEDBACK: qualified_leads must be a list of lead objects."
            
            actual_count = len(task_output['qualified_leads'])
            logger.info(f"Supervisor: Found {actual_count} leads in JSON output")
            
            valid_leads = []
            invalid_leads = []
            
            for lead in task_output['qualified_leads']:
                # Check if this lead's name and company match any in the CSV
                is_valid = any(
                    csv_lead['Name'] == lead.get('lead_name') and 
                    csv_lead['Company Name'] == lead.get('company_name') 
                    for csv_lead in csv_leads
                )
                if is_valid:
                    valid_leads.append(lead)
                else:
                    invalid_leads.append(lead)
                    logger.warning(f"Supervisor: Removing invented lead from JSON: {lead.get('lead_name')} from {lead.get('company_name')}")
            
            # Check for complete failure
            if len(valid_leads) == 0 and len(invalid_leads) > 0:
                logger.error("Supervisor: CRITICAL ERROR - Agent created only fictional leads in JSON format!")
                expected_leads_str = [f"{csv_lead['Name']} ({csv_lead['Company Name']})" for csv_lead in csv_leads[:5]]
                return f"SUPERVISOR_FEEDBACK: CRITICAL ERROR - You completely ignored the CSV file. You must process the {expected_count} leads from {self.csv_file_path}. Expected leads include: {', '.join(expected_leads_str)}."
            
            # Check for missing leads
            if len(valid_leads) < expected_count:
                missing_leads = []
                for csv_lead in csv_leads:
                    if not any(
                        lead.get('lead_name') == csv_lead['Name'] and 
                        lead.get('company_name') == csv_lead['Company Name'] 
                        for lead in valid_leads
                    ):
                        missing_leads.append(f"{csv_lead['Name']} ({csv_lead['Company Name']})")
                
                if missing_leads:
                    return f"SUPERVISOR_FEEDBACK: Missing {len(missing_leads)} leads from CSV: {', '.join(missing_leads[:10])}. Process ALL {expected_count} leads."
            
            # Update with valid leads only
            task_output['qualified_leads'] = valid_leads
            return task_output
        
        # If output format is not recognized for lead validation
        logger.warning("Supervisor: Could not validate lead count - output format not recognized for lead qualification")
        return task_output

    def supervise_and_validate_task_output(self, task_output: Any, original_task: Task) -> str:
        """
        Supervises a task's output by validating it. 
        If valid, returns the JSON output. If invalid, provides feedback for the agent to correct it.
        """
        # First check if this is a lead qualification task and apply special validation
        task_description = original_task.description.lower() if hasattr(original_task, 'description') else ""
        is_lead_qualification = (
            "lead_qualification" in task_description or 
            "qualify" in task_description or 
            (hasattr(original_task, 'name') and "lead_qualification" in original_task.name.lower())
        )
        
        if is_lead_qualification:
            logger.info("Supervisor: Detected lead qualification task - applying CSV lead validation")
            # Apply lead-specific validation
            validated_output = self.validate_lead_qualification_output(task_output)
            
            # If validation returned feedback string, return it for re-delegation
            if isinstance(validated_output, str) and validated_output.startswith("SUPERVISOR_FEEDBACK"):
                return validated_output
            
            # Update task_output with validated version
            task_output = validated_output

        # Step 1: Validate the output structure using TaskValidatorTool
        task_config = self._get_task_config(original_task.description)
        if not task_config or 'expected_output' not in task_config:
            logger.error(f"Supervisor: Could not find task config or expected_output for task: {original_task.description[:50]}...")
            # If we can't find config, we can't validate properly. Return raw output with a warning.
            return f"SUPERVISOR_WARNING: Could not validate due to missing task configuration. Raw output: {str(task_output)}"

        validation_result = self.task_validator_tool.run(
            task_output=json.dumps(task_output, cls=UUIDEncoder) if not isinstance(task_output, str) else task_output, 
            expected_output_description=task_config['expected_output']
        )
        
        # Parse validation_result (it's a string, potentially JSON)
        try:
            validation_data = json.loads(validation_result)
            is_valid = validation_data.get("is_valid", False)
            issues = validation_data.get("issues", [])
        except json.JSONDecodeError:
            is_valid = False
            issues = ["Supervisor: Validation tool returned non-JSON response."]
            logger.error(f"Supervisor: Validation tool returned non-JSON: {validation_result}")

        if not is_valid:
            logger.warning(f"Supervisor: Task output for {original_task.description[:50]}... is invalid. Issues: {issues}. Returning issues for agent to fix.")
            # CrewAI's hierarchical process should handle re-delegation based on this output.
            return f"SUPERVISOR_FEEDBACK: Output is invalid. Please fix the following issues: {'; '.join(issues)}. Ensure your output is valid JSON matching the Pydantic schema: {self._get_pydantic_model_for_task(original_task).__name__ if self._get_pydantic_model_for_task(original_task) else 'details in task description'}."

        # Step 2: If valid JSON, return the validated output
        logger.info(f"Supervisor: Task output for {original_task.description[:50]}... is valid JSON. Returning validated output.")
        try:
            # Return the validated JSON output
            if isinstance(task_output, str):
                try:
                    # Verify it's actually a JSON string and return it formatted
                    parsed_json = json.loads(task_output)
                    return json.dumps(parsed_json, indent=2, cls=UUIDEncoder)
                except json.JSONDecodeError:
                    # If it's a string but not JSON, return as is
                    logger.warning(f"Supervisor: Output for {original_task.description[:50]} was a non-JSON string after validation.")
                    return task_output
            else: # dict, list, Pydantic model
                return json.dumps(task_output, indent=2, cls=UUIDEncoder)
        except Exception as e:
            logger.error(f"Supervisor: Error during output formatting for {original_task.description[:50]}...: {e}. Returning original output.")
            return str(task_output)

    def _get_task_config(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Helper to get task configuration from loaded tasks.yaml based on description."""
        if not self.tasks_config:
            return None
        for _task_name, config in self.tasks_config.items():
            # Match based on the beginning of the task description
            if task_description and config.get('description', '').startswith(task_description[:100]):
                 return config
        return None

    def _get_pydantic_model_for_task(self, task: Task) -> Optional[type[BaseModel]]:
        """Determine the Pydantic model associated with a task."""
        # This mapping needs to be maintained or dynamically determined
        if "lead_qualification_task" in task.config.get('description', '').lower() or (hasattr(task, 'name') and "lead_qualification_task" in task.name.lower()):
            return LeadQualificationResponse
        if "prospect_research_task" in task.config.get('description', '').lower() or (hasattr(task, 'name') and "prospect_research_task" in task.name.lower()):
            return ProspectResearchResponse
        if "email_outreach_task" in task.config.get('description', '').lower() or (hasattr(task, 'name') and "email_outreach_task" in task.name.lower()):
            return EmailOutreachResponse
        if "sales_call_prep_task" in task.config.get('description', '').lower() or (hasattr(task, 'name') and "sales_call_prep_task" in task.name.lower()):
            return SalesCallPrepResponse
        # Add other tasks and their Pydantic models here
        return None

# This class remains for CrewAI to instantiate the agent.
# The actual logic is now in SupervisorAgentLogic and invoked by the supervisor_task.
class SupervisorAgent(Agent):
    logic: SupervisorAgentLogic

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The 'llm', 'agents_config', 'tasks_config' should be passed from SalesGuru class when creating the supervisor agent instance.
        # However, CrewAI's @agent decorator might not pass these directly. We'll rely on them being set if needed by the logic, 
        # or assume the supervisor task will pass necessary context.
        # For now, ensure the tools are correctly assigned as per the crew.py changes.
        self.logic = SupervisorAgentLogic(
            llm=kwargs.get('llm'),
            agents_config=kwargs.get('config', {}).get('agents_config'), # Assuming these might be passed via config
            tasks_config=kwargs.get('config', {}).get('tasks_config'),
            csv_file_path=kwargs.get('config', {}).get('csv_file_path')
        )