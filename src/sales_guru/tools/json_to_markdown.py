import json
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import jsonschema2md

class JSONToMarkdownInput(BaseModel):
    """Input schema for JSONToMarkdownTool."""
    json_data: str = Field(..., description="JSON data to convert to Markdown documentation. Can be a stringified JSON object or array.")
    title: str = Field(default="", description="Optional title for the Markdown document.")

class JSONToMarkdownTool(BaseTool):
    name: str = "JSONToMarkdownConverter"
    description: str = (
        "Converts JSON data into human-readable Markdown documentation. "
        "This tool should only be used when the final output has been verified by a supervisor. "
        "It takes JSON data (either an array of objects or a single object) and converts it into "
        "well-formatted Markdown documentation that is easy to read."
    )
    args_schema: Type[BaseModel] = JSONToMarkdownInput

    def _run(self, json_data: str, title: str = "") -> str:
        """
        Convert JSON data to Markdown documentation.

        Args:
            json_data: String containing JSON data (object or array)
            title: Optional title for the Markdown document

        Returns:
            Markdown formatted documentation
        """
        try:
            # Parse the JSON data
            data = json.loads(json_data)

            # Initialize markdown content
            markdown = ""

            # Add title if provided
            if title:
                markdown += f"# {title}\n\n"

            # Special handling for known container types
            if isinstance(data, dict) and "call_briefs" in data:
                # This is a SalesCallPrepResponse
                markdown += "# Sales Call Briefs\n\n"

                briefs = data["call_briefs"]
                if not briefs:
                    markdown += "No call briefs available.\n"
                    return markdown

                # Process each call brief
                for i, brief in enumerate(briefs):
                    if i > 0:
                        markdown += "\n---\n\n"

                    # Create section for this brief
                    if "lead_name" in brief and "company_name" in brief:
                        markdown += f"## {brief['lead_name']} - {brief['company_name']}\n\n"
                    else:
                        markdown += f"## Call Brief {i+1}\n\n"

                    # Add classification if available
                    if "classification" in brief:
                        markdown += f"**Classification:** {brief['classification']}\n\n"

                    # Process each field in the brief
                    for key, value in brief.items():
                        if key not in ["lead_name", "company_name", "classification"]:
                            # Format field name
                            field_name = key.replace("_", " ").title()

                            # Format different types of values
                            if key == "pain_points" and isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for point in value:
                                    markdown += f"- {point}\n"
                                markdown += "\n"
                            elif key == "talking_points" and isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for point in value:
                                    markdown += f"- {point}\n"
                                markdown += "\n"
                            elif key == "next_steps" and isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for step in value:
                                    markdown += f"- {step}\n"
                                markdown += "\n"
                            elif key == "value_propositions" and isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for prop in value:
                                    markdown += f"- {prop}\n"
                                markdown += "\n"
                            elif key == "objection_responses" and isinstance(value, list):
                                markdown += f"### Objection Handling\n\n"
                                for obj in value:
                                    if isinstance(obj, dict) and "objection" in obj and "response" in obj:
                                        markdown += f"- **{obj['objection']}**: {obj['response']}\n"
                                    else:
                                        markdown += f"- {obj}\n"
                                markdown += "\n"
                            elif isinstance(value, str):
                                if len(value) > 100:  # Long text gets its own section
                                    markdown += f"### {field_name}\n\n{value}\n\n"
                                else:  # Short text gets a simple line
                                    markdown += f"**{field_name}:** {value}\n\n"
                            elif isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for item in value:
                                    markdown += f"- {item}\n"
                                markdown += "\n"
                            elif isinstance(value, dict):
                                markdown += f"### {field_name}\n\n"
                                for k, v in value.items():
                                    markdown += f"- **{k}**: {v}\n"
                                markdown += "\n"
                            else:
                                markdown += f"**{field_name}:** {value}\n\n"

                return markdown

            elif isinstance(data, dict) and "email_templates" in data:
                # This is an EmailOutreachResponse
                markdown += "# Email Outreach Templates\n\n"

                templates = data["email_templates"]
                if not templates:
                    markdown += "No email templates available.\n"
                    return markdown

                # Process each email template
                for i, template in enumerate(templates):
                    if i > 0:
                        markdown += "\n---\n\n"

                    # Create section for this template
                    if "lead_name" in template and "company_name" in template:
                        markdown += f"## {template['lead_name']} - {template['company_name']}\n\n"
                    else:
                        markdown += f"## Email Template {i+1}\n\n"

                    # Add classification if available
                    if "classification" in template:
                        markdown += f"**Classification:** {template['classification']}\n\n"

                    # Add subject line with emphasis
                    if "subject_line" in template:
                        markdown += f"### Subject Line\n\n**{template['subject_line']}**\n\n"

                    # Add email body in a code block for formatting
                    if "email_body" in template:
                        markdown += f"### Email Body\n\n```\n{template['email_body']}\n```\n\n"

                    # Add other fields
                    for key, value in template.items():
                        if key not in ["lead_name", "company_name", "classification", "subject_line", "email_body"]:
                            # Format field name
                            field_name = key.replace("_", " ").title()

                            if key == "ab_test_variations" and isinstance(value, list):
                                markdown += f"### A/B Test Variations\n\n"
                                for variation in value:
                                    if isinstance(variation, dict):
                                        if "element" in variation and "variation" in variation:
                                            markdown += f"- **{variation['element']}**: {variation['variation']}\n"
                                        else:
                                            for k, v in variation.items():
                                                markdown += f"- **{k}**: {v}\n"
                                    else:
                                        markdown += f"- {variation}\n"
                                markdown += "\n"
                            elif isinstance(value, str):
                                markdown += f"**{field_name}:** {value}\n\n"
                            elif isinstance(value, list):
                                markdown += f"### {field_name}\n\n"
                                for item in value:
                                    markdown += f"- {item}\n"
                                markdown += "\n"
                            else:
                                markdown += f"**{field_name}:** {value}\n\n"

                return markdown

            # Use jsonschema2md parser for other cases
            try:
                parser = jsonschema2md.Parser(
                    examples_as_yaml=False,
                    show_examples="all",
                    header_level=1 if title else 0  # Adjust header level if title provided
                )

                # Handle array of objects
                if isinstance(data, list):
                    if not data:
                        return markdown + "# Empty List\n\nNo data provided."

                    # If it's a list of simple objects (like leads), create a better representation
                    if all(isinstance(item, dict) for item in data):
                        # For each item in the list
                        for i, item in enumerate(data):
                            if i > 0:
                                markdown += "\n---\n\n"  # Add separator between items

                            # Create title for each item
                            item_title = self._extract_item_title(item)
                            if item_title:
                                markdown += f"## {item_title}\n\n"
                            else:
                                markdown += f"## Item {i+1}\n\n"

                            # Format each property with description
                            markdown += "### Properties\n\n"
                            for key, value in item.items():
                                formatted_value = self._format_value(value)
                                markdown += f"- **{key}**: {formatted_value}\n"
                            markdown += "\n"

                        return markdown
                    # Use jsonschema2md for complex schema-like objects
                    else:
                        # Try to format it as a schema for jsonschema2md
                        schema = {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                        try:
                            md_lines = parser.parse_schema(schema)
                            markdown += ''.join(md_lines)
                        except Exception:
                            # Fallback to simple formatting
                            markdown += "## Array Content\n\n"
                            markdown += "```json\n"
                            markdown += json.dumps(data, indent=2, ensure_ascii=False)
                            markdown += "\n```\n"

                        return markdown

                # Handle single object with schema-like structure (with properties, etc.)
                elif isinstance(data, dict) and ("properties" in data or "title" in data or "$schema" in data):
                    try:
                        md_lines = parser.parse_schema(data)
                        markdown += ''.join(md_lines)
                        return markdown
                    except:
                        # Fallback to simple formatting
                        pass

                # Handle regular object (non-schema)
                if isinstance(data, dict):
                    # Title handling
                    if "title" in data and not title:
                        markdown += f"# {data['title']}\n\n"

                    # Description handling
                    if "description" in data:
                        markdown += f"_{data['description']}_\n\n"

                    # Format all properties
                    markdown += "## Properties\n\n"
                    for key, value in data.items():
                        if key not in ["title", "description"]:
                            formatted_value = self._format_value(value)
                            markdown += f"- **{key}**: {formatted_value}\n"
                    markdown += "\n"

                    return markdown
            except Exception as e:
                # If jsonschema2md fails, fall back to simpler formatting
                if isinstance(data, dict):
                    markdown += "## Properties\n\n"
                    for key, value in data.items():
                        formatted_value = self._format_value(value)
                        markdown += f"- **{key}**: {formatted_value}\n"
                    markdown += "\n"
                else:
                    # Handle other cases
                    markdown += "Unable to convert to structured markdown. Raw JSON:\n\n"
                    markdown += "```json\n"
                    markdown += json.dumps(data, indent=2, ensure_ascii=False)
                    markdown += "\n```\n"

            return markdown

        except json.JSONDecodeError:
            return "Error: Input is not valid JSON. Please provide valid JSON data."
        except Exception as e:
            return f"Error converting JSON to Markdown: {str(e)}"

    def _extract_item_title(self, item: Dict) -> str:
        """Extract a title from a dictionary item based on common title fields."""
        # Look for common naming fields
        for key in ["name", "title", "lead_name", "company", "company_name",
                   "lead", "person", "contact", "full_name"]:
            if key in item and isinstance(item[key], (str, int, float)):
                if "company" in key.lower() and "name" in item:
                    # If we found company name but there's also a person name, combine them
                    return f"{item['name']} - {item[key]}"
                return str(item[key])

        # If we find both name and company, combine them
        if "name" in item and "company" in item:
            return f"{item['name']} - {item['company']}"

        # If we find lead_name and company_name, combine them
        if "lead_name" in item and "company_name" in item:
            return f"{item['lead_name']} - {item['company_name']}"

        # No good title found
        return ""

    def _format_value(self, value: Any) -> str:
        """Format a value for inclusion in Markdown."""
        if value is None:
            return "_null_"
        elif isinstance(value, bool):
            return "`true`" if value else "`false`"
        elif isinstance(value, (int, float)):
            return f"`{value}`"
        elif isinstance(value, str):
            if len(value) > 100:  # For long text, format as a blockquote
                return f"\n> {value}"
            return value
        elif isinstance(value, list):
            if not value:
                return "_empty list_"
            elif all(isinstance(item, dict) for item in value):
                # For list of objects, create a summary
                return f"_list of {len(value)} objects_"
            else:
                # For simple lists, display inline
                formatted_items = [self._format_simple_value(item) for item in value]
                return ", ".join(formatted_items)
        elif isinstance(value, dict):
            if not value:
                return "_empty object_"
            # For objects, show a summary
            return f"_object with keys: {', '.join(value.keys())}_"
        else:
            return str(value)

    def _format_simple_value(self, value: Any) -> str:
        """Format a simple value for inclusion in a list."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            if len(value) > 30:
                return f'"{value[:27]}..."'
            return f'"{value}"'
        else:
            return str(value)
