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
            input_data = json.loads(input_str)
            
            # Check if required fields are present
            required_fields = ["task_description", "expected_output", "actual_output"]
            for field in required_fields:
                if field not in input_data:
                    return f"Error: Missing required field '{field}' in input"
            
            # Extract the fields
            task_description = input_data["task_description"]
            expected_output = input_data["expected_output"]
            actual_output = input_data["actual_output"]
            
            # Perform basic validation
            validation_result = self._validate_task_output(
                task_description, expected_output, actual_output
            )
            
            # Return the formatted result
            return json.dumps(validation_result, indent=2, cls=UUIDEncoder)
            
        except json.JSONDecodeError:
            return "Error: Input must be valid JSON"
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
        
        # Check for list completeness when ALL is mentioned
        if "ALL" in expected_output:
            if isinstance(actual_output, dict) and any(key in actual_output for key in ["items", "records", "leads", "results"]):
                # Extract the list from the dictionary
                for list_key in ["items", "records", "leads", "results"]:
                    if list_key in actual_output:
                        items = actual_output[list_key]
                        break
            elif isinstance(actual_output, list):
                items = actual_output
            else:
                items = []
                
            if not items:
                result["issues"].append("Expected a list of items but found none")
                result["completeness_score"] = 0.0
                return result
                
            # Extract expected fields
            expected_fields = self._extract_expected_fields(expected_output)
            
            # Check each item for the expected fields
            missing_fields_count = 0
            total_fields_expected = len(expected_fields) * len(items)
            
            for item in items:
                for field in expected_fields:
                    field_found = self._check_field_present(item, field)
                    if not field_found:
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
                
        else:
            # For non-list outputs, do more general validation
            key_phrases = self._extract_key_phrases(expected_output)
            matched_phrases = 0
            
            # Convert actual_output to string for text search if it's not already
            if not isinstance(actual_output, str):
                try:
                    actual_output_str = json.dumps(actual_output, cls=UUIDEncoder)
                except Exception:
                    # Fallback if serialization fails
                    actual_output_str = str(actual_output)
            
            for phrase in key_phrases:
                if phrase.lower() in actual_output_str.lower():
                    matched_phrases += 1
                else:
                    result["missing_elements"].append(phrase)
            
            if len(key_phrases) > 0:
                result["completeness_score"] = matched_phrases / len(key_phrases)
            else:
                result["completeness_score"] = 0.8  # Default if no key phrases
                
            if result["missing_elements"]:
                result["issues"].append(f"Output may be missing key elements: {', '.join(result['missing_elements'])}")
                result["recommendations"].append("Review the expected output format and ensure all elements are included")
        
        # Set validity flag
        result["is_valid"] = result["completeness_score"] >= 0.95
        
        # Add general recommendations
        if not result["is_valid"]:
            if result["completeness_score"] < 0.5:
                result["recommendations"].append("Your output needs significant improvement to meet requirements")
            elif result["completeness_score"] < 0.8:
                result["recommendations"].append("Your output is getting close but still missing important elements")
            else:
                result["recommendations"].append("Your output is almost complete, just missing a few minor elements")
        else:
            result["recommendations"].append("Your output meets all requirements and is ready for submission")
        
        return result
    
    def _extract_expected_fields(self, expected_output: str) -> List[str]:
        """Extract expected fields from the output description."""
        fields = []
        
        # Common field patterns
        if "lead score" in expected_output.lower() or "classification" in expected_output.lower():
            fields.extend(["lead_score", "classification", "reasoning", "value_alignment", "recommended_approach"])
            
        # Look for numbered or bulleted lists in the expected output
        lines = expected_output.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for numbered fields like "1. Field name"
            if line and (line[0].isdigit() and '. ' in line):
                field = line.split('. ', 1)[1].strip().lower()
                # Extract just the field name, not the entire description
                field = field.split(' ')[0].replace(':', '').strip()
                if field and field not in fields:
                    fields.append(field)
                    
            # Check for fields after keywords
            for keyword in ["field", "fields", "include", "includes", "containing", "contains"]:
                if keyword in line.lower():
                    next_line_idx = i + 1
                    if next_line_idx < len(lines):
                        next_line = lines[next_line_idx].strip()
                        if next_line and (next_line[0].isdigit() or next_line[0] == '-' or next_line[0] == '*'):
                            field_part = next_line[1:].strip()
                            if field_part.startswith('. '):
                                field_part = field_part[2:]
                            field = field_part.lower().split(' ')[0].replace(':', '').strip()
                            if field and field not in fields:
                                fields.append(field)
        
        return fields
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for general validation."""
        phrases = []
        
        # Look for phrases after keywords like "must include" or "should contain"
        for keyword in ["must include", "should include", "needs to have", "should contain", "must contain"]:
            if keyword in text.lower():
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    remainder = parts[1].strip()
                    # Extract the next sentence or phrase
                    end_markers = ['.', ',', ';', '\n']
                    end_pos = min((remainder.find(m) for m in end_markers if m in remainder), default=len(remainder))
                    phrase = remainder[:end_pos].strip()
                    if phrase and phrase not in phrases:
                        phrases.append(phrase)
        
        return phrases
    
    def _check_field_present(self, item: Dict, field: str) -> bool:
        """Check if a field is present in an item, handling different formats."""
        # Check various formats of the field name
        possible_keys = [
            field,
            field.replace('_', ' '),
            field.title(),
            field.title().replace('_', ' '),
            field.title().replace('_', ''),
            field.upper(),
            field.lower()
        ]
        
        for key in possible_keys:
            if key in item:
                return True
                
        return False 