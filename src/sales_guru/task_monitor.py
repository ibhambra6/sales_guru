import json
import logging
import uuid
import time
import tiktoken
import os
import csv
from functools import wraps
from typing import Callable, Dict, List, Any, Union
from crewai import Task, Agent
from crewai.task import TaskOutput
import importlib
import random
from pydantic import BaseModel
import yaml
import re

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
        # In newer versions of crewAI, the method structure has changed
        # This patching is no longer needed or compatible
        logger.info("Skipping CrewAI patches as they are no longer compatible with the current version")

        # Previous patching code - kept for reference but commented out
        """
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
        """
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
        truncated_encoded = encoded[:max_tokens - 50]  # Leave room for message
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

        # Load task configurations
        self.tasks_config = self._load_tasks_config()

        # Count available leads
        self.expected_lead_count = self._count_available_leads()

    def _load_tasks_config(self) -> Dict:
        """Load tasks configuration from YAML file"""
        try:
            config_path = os.path.join('src', 'sales_guru', 'config', 'tasks.yaml')
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load tasks config: {e}")
            return {}

    def _count_available_leads(self) -> int:
        """Count the number of leads in the CSV file"""
        try:
            csv_path = os.path.join('knowledge', 'leads.csv')
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                # Skip header row
                next(reader, None)
                # Count remaining rows
                return sum(1 for _ in reader)
        except Exception as e:
            logger.error(f"Failed to count leads: {e}")
            # Default to a reasonable number if we can't count
            return 25

    def _is_tabular_markdown(self, text: str) -> bool:
        """Check if text contains a markdown table"""
        if not text:
            return False

        # Look for pattern: | header1 | header2 | etc with a separator row
        table_pattern = r"\|[^\|]+\|[^\|]+\|.*\n\|[\s-]+\|[\s-]+\|"
        return bool(re.search(table_pattern, text))

    def _wrap_markdown_tables(self, text: str, wrap_width: int = 80) -> str:
        """Process markdown text to wrap long lines in tables"""
        from sales_guru.schemas import wrap_text_for_markdown_table

        if not self._is_tabular_markdown(text):
            return text

        lines = text.split("\n")
        in_table = False
        header_row = []
        separator_row = ""
        data_rows = []
        result_lines = []

        for line in lines:
            # Check if this is a table row
            if line.strip().startswith("|") and line.strip().endswith("|"):
                # If this is the first table row we've seen, it's the header
                if not in_table:
                    in_table = True
                    header_row = [cell.strip() for cell in line.strip().strip("|").split("|")]
                    result_lines.append(line)
                # If this is the second table row and it contains only dashes and pipes, it's the separator
                elif separator_row == "" and all(c in "-| " for c in line):
                    separator_row = line
                    result_lines.append(line)
                # Otherwise it's a data row
                else:
                    # Process cells in this row to wrap long text
                    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                    wrapped_cells = [wrap_text_for_markdown_table(cell, wrap_width) for cell in cells]
                    wrapped_row = "| " + " | ".join(wrapped_cells) + " |"
                    result_lines.append(wrapped_row)
            else:
                # Not a table row, reset in_table flag if we were in a table
                if in_table:
                    in_table = False
                    # Reset table tracking variables
                    header_row = []
                    separator_row = ""
                    data_rows = []
                result_lines.append(line)

        return "\n".join(result_lines)

    def _validate_lead_count(self, output: Any) -> Any:
        """
        Validate that the lead qualification output only contains the expected number of leads.
        If there are too many leads, truncate to match the expected count.
        If there are too few leads, log a warning but allow the process to continue.

        Args:
            output: The output to validate

        Returns:
            The validated output with the correct number of leads
        """
        # Get the CSV leads for validation
        csv_leads = self._get_csv_lead_data()
        expected_count = len(csv_leads)

        # For Pydantic models with qualified_leads attribute
        if hasattr(output, 'qualified_leads') and isinstance(output.qualified_leads, list):
            # actual_count = len(output.qualified_leads)  # Not used in current logic

            # First, check if any of the leads match the CSV data
            valid_leads = []
            invalid_leads = []

            for lead in output.qualified_leads:
                # Check if this lead's name and company match any in the CSV
                is_valid = any(csv_lead['Name'] == lead.lead_name and csv_lead['Company Name'] == lead.company_name
                              for csv_lead in csv_leads)
                if is_valid:
                    valid_leads.append(lead)
                else:
                    invalid_leads.append(lead)
                    logger.warning(f"Removing invented lead: {lead.lead_name} from {lead.company_name}")

            # If we have NO valid leads, this means the agent completely ignored the CSV
            if len(valid_leads) == 0 and len(invalid_leads) > 0:
                logger.error("Lead qualification agent completely ignored the CSV file and created fictional leads!")

                expected_leads_str = [f"{csv_lead['Name']} ({csv_lead['Company Name']})" for csv_lead in csv_leads[:5]]
                actual_leads_str = [f"{lead.lead_name} ({lead.company_name})" for lead in invalid_leads[:5]]

                logger.error(f"Expected leads from CSV: {expected_leads_str}")
                logger.error(f"Actual leads created: {actual_leads_str}")

                # Return empty results with an error message
                from sales_guru.schemas import LeadQualification
                error_lead = LeadQualification(
                    lead_name="ERROR",
                    company_name="Agent created fictional leads instead of processing CSV",
                    email_address="error@example.com",
                    phone_number="000-000-0000",
                    lead_score=0,
                    classification="COLD",
                    reasoning="The lead qualification agent failed to process the actual leads from the CSV file and instead created fictional leads. Please check the agent configuration and CSV processing logic.",
                    value_alignment="No value - this is an error condition",
                    recommended_approach="Fix the lead qualification agent to properly read and process the CSV file"
                )
                output.qualified_leads = [error_lead]
                return output

            # If we have some valid leads but fewer than expected, some CSV leads weren't processed
            if len(valid_leads) < expected_count:
                logger.warning(f"Only found {len(valid_leads)} valid leads out of {expected_count} expected. Some CSV leads may not have been processed.")
                missing_leads = []
                for csv_lead in csv_leads:
                    if not any(lead.lead_name == csv_lead['Name'] and lead.company_name == csv_lead['Company Name']
                              for lead in valid_leads):
                        missing_leads.append(f"{csv_lead['Name']} ({csv_lead['Company Name']})")

                if missing_leads:
                    logger.warning(f"Missing leads from CSV: {missing_leads[:5]}")

            # Update the output with only valid leads
            output.qualified_leads = valid_leads
            return output

        # For markdown table outputs
        elif isinstance(output, str) and self._is_tabular_markdown(output):
            # Parse the table to count the leads
            lines = output.strip().split('\n')
            if len(lines) < 3:  # Need header, separator, and at least one data row
                return output

            header = lines[0]
            separator = lines[1]
            data_rows = lines[2:]

            valid_rows = []
            invalid_rows = []

            for row in data_rows:
                cells = [cell.strip() for cell in row.strip().strip('|').split('|')]

                # Check structure matches our expected table format
                if len(cells) < 2:
                    continue

                # Try to find name and company in the row
                # Cells[0] should be Name and Cells[1] should be Company in our table structure
                lead_name = cells[0].strip()
                company_name = cells[1].strip()

                # Check if this lead's name and company match any in the CSV
                is_valid = any(csv_lead['Name'] == lead_name and csv_lead['Company Name'] == company_name
                              for csv_lead in csv_leads)
                if is_valid:
                    valid_rows.append(row)
                else:
                    invalid_rows.append(row)
                    logger.warning(f"Removing invented lead from table: {lead_name} from {company_name}")

            # If no valid rows found but invalid ones exist, the agent ignored the CSV
            if len(valid_rows) == 0 and len(invalid_rows) > 0:
                logger.error("Lead qualification agent completely ignored the CSV file and created fictional leads in table format!")
                # Create an error table
                error_row = "| ERROR | Agent created fictional leads instead of processing CSV | error@example.com | 000-000-0000 | 0 | COLD | Agent failed to process CSV file | No value | Fix agent configuration |"
                return f"{header}\n{separator}\n{error_row}"

            # If we have fewer valid rows than expected, log a warning
            if len(valid_rows) < expected_count:
                logger.warning(f"Only found {len(valid_rows)} valid leads out of {expected_count} expected in table. Some CSV leads may not have been processed.")

            # Create new table with only valid rows
            if len(valid_rows) > 0:
                truncated_table = [header, separator] + valid_rows
                return '\n'.join(truncated_table)
            else:
                return output

        # For dict/JSON responses that might have a leads array
        elif isinstance(output, dict) and 'qualified_leads' in output and isinstance(output['qualified_leads'], list):
            # actual_count = len(output['qualified_leads'])  # Not used in current logic

            valid_leads = []
            invalid_leads = []

            for lead in output['qualified_leads']:
                # Check if this lead's name and company match any in the CSV
                is_valid = any(csv_lead['Name'] == lead.get('lead_name') and csv_lead['Company Name'] == lead.get('company_name')
                              for csv_lead in csv_leads)
                if is_valid:
                    valid_leads.append(lead)
                else:
                    invalid_leads.append(lead)
                    logger.warning(f"Removing invented lead from JSON: {lead.get('lead_name')} from {lead.get('company_name')}")

            # If no valid leads found but invalid ones exist, the agent ignored the CSV
            if len(valid_leads) == 0 and len(invalid_leads) > 0:
                logger.error("Lead qualification agent completely ignored the CSV file and created fictional leads in JSON format!")
                # Create an error lead
                error_lead = {
                    "lead_name": "ERROR",
                    "company_name": "Agent created fictional leads instead of processing CSV",
                    "email_address": "error@example.com",
                    "phone_number": "000-000-0000",
                    "lead_score": 0,
                    "classification": "COLD",
                    "reasoning": "Agent failed to process CSV file",
                    "value_alignment": "No value",
                    "recommended_approach": "Fix agent configuration"
                }
                output['qualified_leads'] = [error_lead]
                return output

            # Update the output with only valid leads
            output['qualified_leads'] = valid_leads

        return output

    def _get_csv_lead_data(self) -> List[Dict[str, str]]:
        """Get lead data from the CSV file"""
        csv_leads = []
        try:
            csv_path = os.path.join('knowledge', 'leads.csv')
            with open(csv_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    csv_leads.append(row)
            return csv_leads
        except Exception as e:
            logger.error(f"Failed to read leads CSV: {e}")
            return []

    def _format_markdown_table(self, markdown_table: str) -> str:
        """
        Ensure the markdown table is properly formatted with consistent spacing and alignment.

        Args:
            markdown_table: Raw markdown table string

        Returns:
            Properly formatted markdown table
        """
        if not markdown_table or not self._is_tabular_markdown(markdown_table):
            return markdown_table

        lines = markdown_table.strip().split('\n')
        if len(lines) < 3:  # Need header, separator, and at least one data row
            return markdown_table

        # Extract parts of the table
        header = lines[0]
        separator_line = lines[1]
        data_rows = lines[2:]

        # Parse header cells
        header_cells = []
        if header.startswith('|'):
            header = header[1:]
        if header.endswith('|'):
            header = header[:-1]
        header_cells = [cell.strip() for cell in header.split('|')]

        # Calculate column widths based on content
        col_widths = [len(cell) for cell in header_cells]

        # Analyze data rows to determine column widths
        for row in data_rows:
            if row.startswith('|'):
                row = row[1:]
            if row.endswith('|'):
                row = row[:-1]
            cells = [cell.strip() for cell in row.split('|')]

            # Update column widths if needed
            for i, cell in enumerate(cells):
                if i < len(col_widths):
                    # For cells with <br>, calculate width based on the longest line
                    if '<br>' in cell:
                        max_line_length = max(len(line) for line in cell.split('<br>'))
                        col_widths[i] = max(col_widths[i], max_line_length)
                    else:
                        col_widths[i] = max(col_widths[i], len(cell))

        # Rebuild header with proper spacing
        formatted_header = '| ' + ' | '.join(header_cells[i].ljust(col_widths[i])
                                         for i in range(len(header_cells))) + ' |'

        # Rebuild separator with proper spacing
        formatted_separator = '| ' + ' | '.join('-' * col_widths[i] for i in range(len(col_widths))) + ' |'

        # Rebuild data rows with proper spacing
        formatted_rows = []
        for row in data_rows:
            if row.startswith('|'):
                row = row[1:]
            if row.endswith('|'):
                row = row[:-1]
            cells = [cell.strip() for cell in row.split('|')]

            # Format each cell, handling <br> tags by ensuring each line is properly formatted
            formatted_cells = []
            for i, cell in enumerate(cells):
                if i < len(col_widths):
                    if '<br>' in cell:
                        lines = cell.split('<br>')
                        # Format the first line normally
                        formatted_cell = lines[0].ljust(col_widths[i])
                        # For remaining lines, add appropriate padding and <br> tags
                        for line in lines[1:]:
                            formatted_cell += '<br>' + line.ljust(col_widths[i])
                        formatted_cells.append(formatted_cell)
                    else:
                        formatted_cells.append(cell.ljust(col_widths[i]))
                else:
                    formatted_cells.append(cell)  # Extra cell, keep as is

            formatted_row = '| ' + ' | '.join(formatted_cells) + ' |'
            formatted_rows.append(formatted_row)

        # Combine everything into a properly formatted table
        return '\n'.join([formatted_header, formatted_separator] + formatted_rows)

    def save_processed_output(self, task: Task, output: Any) -> None:
        """
        Save the task output to the specified file with proper formatting for markdown.

        Args:
            task: The task that produced the output
            output: The output data
        """
        # Find the task config
        task_config = None
        task_id = task.id

        # Extract task name from the task
        task_name = None
        for name, config in self.tasks_config.items():
            if task.description and task.description.startswith(config.get('description', '').split('\n')[0][:50]):
                task_name = name
                task_config = config
                break

        if not task_config or 'output_file' not in task_config:
            logger.warning(f"No output file specified for task {task_id}")
            return

        output_file = task_config['output_file']

        # For lead qualification task, validate the lead count
        if task_name == 'lead_qualification_task':
            output = self._validate_lead_count(output)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert output to string if it's a Pydantic model with to_markdown_table method
        output_text = ""
        if hasattr(output, 'to_markdown_table'):
            # Get JSON representation that will be converted to markdown
            json_data = output.to_markdown_table(wrap_width=80)

            # Check if the result is already formatted markdown (error message)
            if json_data.startswith('#'):
                output_text = json_data
            else:
                # Try to convert JSON to markdown using jsonschema2md
                try:
                    import jsonschema2md

                    # Parse the JSON data
                    data = json.loads(json_data)

                    # Detect the title for the markdown document
                    title = ""
                    if task_name == 'email_outreach_task':
                        title = "Email Outreach Templates"
                    elif task_name == 'sales_call_prep_task':
                        title = "Sales Call Briefs"
                    elif task_name == 'lead_qualification_task':
                        title = "Lead Qualification Results"
                    elif task_name == 'prospect_research_task':
                        title = "Prospect Research Results"

                    # Initialize the parser
                    parser = jsonschema2md.Parser(
                        examples_as_yaml=False,
                        show_examples="all",
                        header_level=0
                    )

                    # Create schema-like structure if needed
                    if isinstance(data, dict) and any(key in data for key in ["email_templates", "call_briefs", "qualified_leads", "enriched_leads"]):
                        # Add a title to the output
                        output_text = f"# {title}\n\n"

                        # Process each item category
                        for category, items in data.items():
                            if isinstance(items, list) and len(items) > 0:
                                # Add a section for this category
                                category_title = category.replace('_', ' ').title()
                                output_text += f"## {category_title}\n\n"

                                # Process each item in the list
                                for i, item in enumerate(items):
                                    if i > 0:
                                        output_text += "\n---\n\n"

                                    # Try to extract a title for this item
                                    item_title = ""
                                    if "lead_name" in item and "company_name" in item:
                                        item_title = f"{item['lead_name']} - {item['company_name']}"
                                    elif "name" in item:
                                        item_title = item["name"]

                                    if item_title:
                                        output_text += f"### {item_title}\n\n"
                                    else:
                                        output_text += f"### Item {i+1}\n\n"

                                    # Process the item's properties
                                    for key, value in item.items():
                                        if key not in ["lead_name", "company_name"] or not item_title:
                                            # Format the key
                                            formatted_key = key.replace('_', ' ').title()

                                            # Format the value
                                            if isinstance(value, list):
                                                if all(isinstance(v, dict) for v in value):
                                                    # For lists of objects
                                                    output_text += f"#### {formatted_key}\n\n"
                                                    for j, obj in enumerate(value):
                                                        for k, v in obj.items():
                                                            k_formatted = k.replace('_', ' ').title()
                                                            output_text += f"- **{k_formatted}**: {v}\n"
                                                        if j < len(value) - 1:
                                                            output_text += "\n"
                                            else:
                                                # For lists of simple values
                                                output_text += f"#### {formatted_key}\n\n"
                                                for val in value:
                                                    output_text += f"- {val}\n"
                                            output_text += "\n"
                                        elif isinstance(value, dict):
                                            # For dictionaries
                                            output_text += f"#### {formatted_key}\n\n"
                                            for k, v in value.items():
                                                k_formatted = k.replace('_', ' ').title()
                                                output_text += f"- **{k_formatted}**: {v}\n"
                                            output_text += "\n"
                                        elif isinstance(value, str) and len(value) > 100:
                                            # For long text
                                            output_text += f"#### {formatted_key}\n\n{value}\n\n"
                                        else:
                                            # For simple values
                                            output_text += f"#### {formatted_key}\n\n{value}\n\n"
                    else:
                        # Use the parser directly if possible
                        try:
                            md_lines = parser.parse_schema(data)
                            output_text = f"# {title}\n\n" + ''.join(md_lines)
                        except Exception as e:
                            # Fallback to simple formatting
                            output_text = f"# {title}\n\n"
                            output_text += "```json\n"
                            output_text += json.dumps(data, indent=2, ensure_ascii=False)
                            output_text += "\n```\n"

                except Exception as e:
                    logger.warning(f"Failed to convert JSON to markdown: {e}")
                    # Fallback to the original JSON
                    output_text = f"# {title}\n\n"
                    output_text += "```json\n"
                    output_text += json_data
                    output_text += "\n```\n"
        elif hasattr(output, 'model_dump_json'):
            # It's a Pydantic model but doesn't have to_markdown_table
            try:
                # Try to get module path from the model class
                model_module = output.__class__.__module__
                model_name = output.__class__.__name__

                # Import the module dynamically
                module = importlib.import_module(model_module)

                # Get the container class that might have to_markdown_table method
                container_class = None
                if hasattr(module, f"{model_name}Response"):
                    container_class = getattr(module, f"{model_name}Response")
                    container = container_class(**{f"{model_name.lower()}s": [output]})
                    if hasattr(container, 'to_markdown_table'):
                        json_data = container.to_markdown_table(wrap_width=80)
                        # Process the JSON data as above
                        # (Same code as above for processing JSON to markdown)
                        if json_data.startswith('#'):
                            output_text = json_data
                        else:
                            # Convert JSON to markdown
                            # ... (similar to the logic above)
                            output_text = f"# Model Output\n\n```json\n{json_data}\n```\n"

                if not output_text:
                    # Fallback to JSON representation
                    output_text = f"# {model_name} Output\n\n```json\n{output.model_dump_json(indent=2)}\n```\n"
            except Exception as e:
                logger.warning(f"Failed to convert Pydantic model to markdown: {e}")
                # Fallback to JSON representation
                output_text = f"# Model Output\n\n```json\n{output.model_dump_json(indent=2)}\n```\n"
        elif isinstance(output, str):
            # If it's already markdown, use it directly
            if output.startswith('#') or output.startswith('```') or '|' in output and '-|-' in output:
                output_text = output
            else:
                # Plain text output - format as markdown
                output_text = f"# Task Output\n\n{output}\n"
        elif isinstance(output, (dict, list)):
            # JSON-serializable objects - convert to markdown
            try:
                import jsonschema2md

                # Determine a title
                title = "JSON Data"
                if task_name:
                    title = task_name.replace('_', ' ').title()

                # Initialize the parser
                parser = jsonschema2md.Parser(
                    examples_as_yaml=False,
                    show_examples="all",
                    header_level=0
                )

                # Try to use the parser
                try:
                    md_lines = parser.parse_schema(output)
                    output_text = f"# {title}\n\n" + ''.join(md_lines)
                except Exception:
                    # Fallback to JSON formatting
                    output_text = f"# {title}\n\n```json\n{json.dumps(output, indent=2, cls=UUIDEncoder)}\n```\n"
            except ImportError:
                # If jsonschema2md is not available, fall back to JSON
                output_text = f"# JSON Output\n\n```json\n{json.dumps(output, indent=2, cls=UUIDEncoder)}\n```\n"
        else:
            # Anything else, convert to string and format as markdown
            output_text = f"# Task Output\n\n```\n{str(output)}\n```\n"

        # Write the output to the file
        try:
            with open(output_file, 'w') as file:
                file.write(output_text)
            logger.info(f"Saved processed output to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output to {output_file}: {e}")

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
                    return False  # Retry
                else:
                    logger.error(f"Task {task_id} - Empty result after {attempt_number} attempts. Giving up.")
                    return True  # Accept and continue to prevent infinite loops

            # Save the processed output to the specified file
            self.save_processed_output(task, result_value)

            # Successful execution
            return True

        return callback

    def _extract_output_value(self, output: Any) -> Any:
        """Extract the actual output value from a TaskOutput object or other containers"""
        if hasattr(output, "raw_output"):
            return output.raw_output
        elif hasattr(output, "output") and output.output:
            return output.output
        return output

    def _validate_and_repair_json_output(self, task: Task, output: Any) -> Any:
        """
        Validate and attempt to repair JSON output for downstream tasks.

        Args:
            task: The task that produced the output
            output: The output to validate and repair

        Returns:
            The validated and possibly repaired output
        """
        # Only process if the task is email_outreach_task or sales_call_prep_task
        if not hasattr(task, 'id') or task.id not in ['email_outreach_task', 'sales_call_prep_task']:
            return output

        # If the output is already a valid model instance, return it
        if hasattr(output, 'model_dump') and callable(output.model_dump):
            return output

        # If the output is a string, try to parse it as JSON
        if isinstance(output, str):
            try:
                json_data = json.loads(output)

                # Check if the structure matches what we expect
                if task.id == 'email_outreach_task':
                    from sales_guru.schemas import EmailOutreachResponse, EmailTemplate

                    # Check if it's already the right structure
                    if isinstance(json_data, dict) and 'email_templates' in json_data:
                        return EmailOutreachResponse(**json_data)

                    # If it's a list of dicts, try to convert to EmailTemplate objects
                    if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
                        templates = []
                        for item in json_data:
                            # Ensure all required fields are present with default values if needed
                            item.setdefault('lead_name', 'Unknown Lead')
                            item.setdefault('company_name', 'Unknown Company')
                            item.setdefault('classification', 'WARM')  # Default to WARM
                            item.setdefault('subject_line', 'No subject provided')
                            item.setdefault('email_body', 'No email body provided')
                            item.setdefault('follow_up_timing', 'Not specified')
                            item.setdefault('alternative_contact_channels', None)
                            item.setdefault('ab_test_variations', [])

                            templates.append(EmailTemplate(**item))

                        return EmailOutreachResponse(email_templates=templates)

                elif task.id == 'sales_call_prep_task':
                    from sales_guru.schemas import SalesCallPrepResponse, CallBrief

                    # Check if it's already the right structure
                    if isinstance(json_data, dict) and 'call_briefs' in json_data:
                        return SalesCallPrepResponse(**json_data)

                    # If it's a list of dicts, try to convert to CallBrief objects
                    if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
                        briefs = []
                        for item in json_data:
                            # Ensure all required fields are present with default values if needed
                            item.setdefault('lead_name', 'Unknown Lead')
                            item.setdefault('company_name', 'Unknown Company')
                            item.setdefault('classification', 'WARM')  # Default to WARM
                            item.setdefault('company_snapshot', 'No company information provided')
                            item.setdefault('decision_maker_profile', 'No decision-maker information provided')
                            item.setdefault('relationship_history', 'No previous interactions')
                            item.setdefault('pain_points', ['No pain points identified'])
                            item.setdefault('talking_points', ['No talking points prepared'])
                            item.setdefault('objection_responses', [])
                            item.setdefault('next_steps', ['No next steps defined'])
                            item.setdefault('recent_developments', 'No recent developments noted')
                            item.setdefault('competitive_insights', None)
                            item.setdefault('value_propositions', ['No value propositions defined'])

                            briefs.append(CallBrief(**item))

                        return SalesCallPrepResponse(call_briefs=briefs)
            except Exception as e:
                logger.warning(f"Failed to parse or repair JSON output: {e}")

        # If we couldn't repair it, return the original output
        return output

    def enhance_task(self, task: Task) -> Task:
        """
        Enhance a task with completion guarantees and validation.

        Args:
            task: The task to enhance

        Returns:
            The enhanced task
        """
        # Initialize task attempt counter
        task_id = getattr(task, 'id', f"task_{uuid.uuid4()}")
        self.task_attempt_counter[task_id] = 0
        self.task_history[task_id] = []

        # Create monitoring callback
        monitor_callback = self.create_task_callback(task)

        # Store original callback if it exists
        original_callback = task.callback if callable(task.callback) else None

        # Create a new callback that combines all functionality
        def combined_callback(output):
            """Combined callback that handles validation, retries, and output saving"""
            # Extract the actual output value first
            result_value = self._extract_output_value(output)

            # For lead qualification task, validate the lead count regardless of whether it has an output file
            if hasattr(task, 'id') and str(task.id) == 'lead_qualification_task':
                logger.info("Applying lead qualification validation...")
                result_value = self._validate_lead_count(result_value)

                # Update the output object with the validated result
                if hasattr(output, 'raw_output'):
                    output.raw_output = result_value
                elif hasattr(output, 'output'):
                    output.output = result_value
                else:
                    output = result_value

                # Log the validation result
                if hasattr(result_value, 'qualified_leads'):
                    logger.info(f"Lead qualification validation complete. Output contains {len(result_value.qualified_leads)} leads.")
                    if result_value.qualified_leads and result_value.qualified_leads[0].lead_name == "ERROR":
                        logger.error("Lead qualification failed validation - agent created fictional leads!")
                    else:
                        logger.info("Lead qualification passed validation.")

            # Repair JSON output if necessary (for other task types)
            if hasattr(task, 'id') and str(task.id) not in ['lead_qualification_task']:
                output = self._validate_and_repair_json_output(task, output)

            # Log and save the output (only for tasks with output files)
            self.save_processed_output(task, output)

            # Call the monitoring callback
            monitor_result = monitor_callback(output)

            # Call the original callback if it exists
            if original_callback:
                original_result = original_callback(output)
                return monitor_result and original_result

            # Default to allowing task to proceed
            return monitor_result

        # Set the combined callback
        task.callback = combined_callback

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
