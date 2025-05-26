import json
import uuid
from typing import Dict, List, Union
from crewai.tools import BaseTool

# Custom JSON encoder that can handle UUID objects
class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            # Convert UUID to string
            return str(obj)
        return super().default(obj)

class TaskValidatorTool(BaseTool):
    """Tool for validating task completion and ensuring all requirements are met."""

    name: str = "TaskValidator"
    description: str = """
    Use this tool to validate if your task output meets all requirements before submitting.
    Input should be a JSON object with these fields:
    - task_description: A description of what your task requires
    - expected_output: What format/content is expected in your output
    - actual_output: Your current output that needs validation

    The tool will check if your output meets all requirements and provide feedback on what's missing.
    """

    def _run(self, input_str: str) -> str:
        """Run the task validation."""
        try:
            # Parse the input as JSON
            try:
                input_data = json.loads(input_str)
            except json.JSONDecodeError:
                return "Error: Input must be valid JSON"

            # Check if required fields are present
            required_fields = ["task_description", "expected_output", "actual_output"]
            missing_fields = []
            for field in required_fields:
                if field not in input_data:
                    missing_fields.append(field)

            if missing_fields:
                return f"Error: Missing required fields: {', '.join(missing_fields)}"

            # Extract the fields
            task_description = input_data["task_description"]
            expected_output = input_data["expected_output"]
            actual_output = input_data["actual_output"]

            # Perform validation
            validation_result = self._validate_task_output(
                task_description, expected_output, actual_output
            )

            # Return the formatted result
            return json.dumps(validation_result, indent=2, cls=UUIDEncoder)

        except Exception as e:
            return f"Error during validation: {str(e)}"

    def _validate_task_output(
        self,
        task_description: str,
        expected_output: str,
        actual_output: Union[str, Dict, List]
    ) -> Dict:
        """
        Validate the task output against expectations.

        Args:
            task_description: Description of the task
            expected_output: Expected output format/content
            actual_output: Actual output provided

        Returns:
            Dict with validation results
        """
        result = {
            "is_valid": False,
            "completeness_score": 0.0,
            "issues": [],
            "missing_elements": [],
            "recommendations": []
        }

        # Check if output exists
        if not actual_output:
            result["issues"].append("Empty or null output")
            result["recommendations"].append("Ensure your output contains all required information")
            return result

        # If actual_output is a string but expected JSON, try to parse it
        if isinstance(actual_output, str) and "JSON" in expected_output:
            try:
                actual_output = json.loads(actual_output)
            except json.JSONDecodeError:
                result["issues"].append("Output is not valid JSON as expected")
                result["recommendations"].append("Ensure your output is formatted as valid JSON")
                result["completeness_score"] = 0.1
                return result

        # Determine if we need to validate a list structure
        if self._is_list_validation_required(expected_output):
            self._validate_list_structure(result, actual_output, expected_output)
        else:
            self._validate_general_content(result, actual_output, expected_output)

        # Set validity flag
        result["is_valid"] = result["completeness_score"] >= 0.95

        # Add appropriate recommendations
        self._add_recommendations(result)

        return result

    def _is_list_validation_required(self, expected_output: str) -> bool:
        """Determine if the expected output requires list validation."""
        return "ALL" in expected_output or any(x in expected_output.lower() for x in ["list", "array", "items"])

    def _validate_list_structure(self, result: Dict, output: Union[Dict, List], expected_output: str) -> None:
        """Validate output that is expected to contain a list of items."""
        # Extract the list items
        items = self._extract_list_items(output)

        if not items:
            result["issues"].append("Expected a list of items but found none")
            result["completeness_score"] = 0.0
            return

        # Extract expected fields
        expected_fields = self._extract_expected_fields(expected_output)

        # Check each item for the expected fields
        missing_fields_count = 0
        total_fields_expected = len(expected_fields) * len(items)

        for item in items:
            for field in expected_fields:
                if not self._check_field_present(item, field):
                    missing_fields_count += 1
                    if field not in result["missing_elements"]:
                        result["missing_elements"].append(field)

        if missing_fields_count > 0:
            result["issues"].append(f"Missing {missing_fields_count} field occurrences across {len(items)} items")
            result["recommendations"].append("Ensure all items have all required fields")

        # Calculate completeness score
        if total_fields_expected > 0:
            fields_present = total_fields_expected - missing_fields_count
            result["completeness_score"] = fields_present / total_fields_expected
        else:
            result["completeness_score"] = 0.5  # Default if we can't calculate

    def _extract_list_items(self, output: Union[Dict, List]) -> List:
        """Extract the list items from the output."""
        if isinstance(output, dict):
            # Try to find the list in common container fields
            for list_key in ["items", "records", "leads", "results", "data"]:
                if list_key in output and isinstance(output[list_key], list):
                    return output[list_key]

            # If no list found in standard keys, try to find any list value
            for value in output.values():
                if isinstance(value, list) and len(value) > 0:
                    return value

            return []
        elif isinstance(output, list):
            return output
        else:
            return []

    def _validate_general_content(self, result: Dict, output: Union[str, Dict, List], expected_output: str) -> None:
        """Validate general (non-list) output content."""
        # Extract key phrases that should be present
        key_phrases = self._extract_key_phrases(expected_output)
        matched_phrases = 0

        # Convert output to string for text matching if it's not already
        if not isinstance(output, str):
            try:
                output_str = json.dumps(output, cls=UUIDEncoder)
            except Exception:
                # Fallback if serialization fails
                output_str = str(output)
        else:
            output_str = output

        # Check for each key phrase
        for phrase in key_phrases:
            if phrase.lower() in output_str.lower():
                matched_phrases += 1
            else:
                result["missing_elements"].append(phrase)

        # Calculate completeness score
        if len(key_phrases) > 0:
            result["completeness_score"] = matched_phrases / len(key_phrases)
        else:
            result["completeness_score"] = 0.8  # Default if no key phrases

        # Add issues if elements are missing
        if result["missing_elements"]:
            result["issues"].append(f"Output may be missing key elements: {', '.join(result['missing_elements'])}")
            result["recommendations"].append("Review the expected output format and ensure all elements are included")

    def _add_recommendations(self, result: Dict) -> None:
        """Add appropriate recommendations based on completeness score."""
        # Clear any existing recommendations to avoid duplicates
        if not result["is_valid"]:
            if result["completeness_score"] < 0.5:
                result["recommendations"].append("Your output needs significant improvement to meet requirements")
            elif result["completeness_score"] < 0.8:
                result["recommendations"].append("Your output is getting close but still missing important elements")
            else:
                result["recommendations"].append("Your output is almost complete, just missing a few minor elements")
        else:
            result["recommendations"].append("Your output meets all requirements and is ready for submission")

    def _extract_expected_fields(self, expected_output: str) -> List[str]:
        """Extract expected fields from the output description."""
        fields = []

        # Email Outreach specific fields
        if "email" in expected_output.lower() and ("outreach" in expected_output.lower() or "template" in expected_output.lower()):
            fields.extend([
                "lead_name", "company", "classification", "subject_line", "email_body",
                "follow_up_timing", "alternative_contact_channels", "ab_test_variations"
            ])

        # Sales Call Prep specific fields
        if "call" in expected_output.lower() and ("brief" in expected_output.lower() or "prep" in expected_output.lower()):
            fields.extend([
                "lead_name", "company", "classification", "company_snapshot", "decision_maker_profile",
                "relationship_history", "pain_points", "talking_points", "objection_responses",
                "next_steps", "recent_developments", "competitive_insights", "value_propositions"
            ])

        # Lead Qualification specific fields
        if "lead score" in expected_output.lower() or "classification" in expected_output.lower():
            fields.extend(["lead_name", "company_name", "email_address", "phone_number",
                          "lead_score", "classification", "reasoning", "value_alignment", "recommended_approach"])

        # Look for numbered or bulleted lists in the expected output
        lines = expected_output.split('\n')
        for line in lines:
            line = line.strip()

            # Check for numbered fields like "1. Field name" or bulleted lists
            is_list_item = (
                line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                line.startswith(('- ', '* ')) or
                line.startswith(('•'))
            )

            if is_list_item:
                # Extract the field name
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    field_part = parts[1].strip()

                    # Look for a field name in quotes
                    if '"' in field_part:
                        quoted_part = field_part.split('"')[1].strip()
                        fields.append(quoted_part.lower().replace(' ', '_'))
                    elif '`' in field_part:
                        # Look for fields in backticks
                        quoted_part = field_part.split('`')[1].strip()
                        fields.append(quoted_part.lower().replace(' ', '_'))
                    else:
                        # Otherwise just use the first few words
                        field_name = ' '.join(field_part.split()[:3]).lower().replace(' ', '_')
                        fields.append(field_name)

        # Return unique fields
        return list(set(fields))

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from the expected output description."""
        phrases = []

        # Certain standard phrases that should be included
        if "summary" in text.lower():
            phrases.append("summary")
        if "recommendation" in text.lower():
            phrases.append("recommendation")
        if "conclusion" in text.lower():
            phrases.append("conclusion")
        if "analysis" in text.lower():
            phrases.append("analysis")

        # Extract phrases from bullet points and numbers
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*', '•')):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    phrases.append(parts[1].strip().split(':')[0].strip())

        return phrases

    def _check_field_present(self, item: Dict, field: str) -> bool:
        """Check if a field is present in an item."""
        if not isinstance(item, dict):
            return False

        # Handle nested fields (e.g., "contact.email")
        if '.' in field:
            parts = field.split('.', 1)
            if parts[0] in item and isinstance(item[parts[0]], dict):
                return self._check_field_present(item[parts[0]], parts[1])
            return False

        # Check direct field
        if field in item:
            # Consider empty strings as not present, but allow False and 0
            return bool(item[field]) or item[field] == 0 or item[field] is False

        # Try alternative field naming conventions
        field_variants = [
            field,
            field.lower(),  # Lowercase
            field.upper(),  # Uppercase
            '_'.join(w for w in field.split('_') if w),  # Remove extra underscores
            ''.join(w.capitalize() for w in field.split('_')),  # CamelCase
            ''.join(w.capitalize() if i > 0 else w for i, w in enumerate(field.split('_'))),  # camelCase
        ]

        for variant in field_variants:
            if variant in item:
                return bool(item[variant]) or item[variant] == 0 or item[variant] is False

        return False
