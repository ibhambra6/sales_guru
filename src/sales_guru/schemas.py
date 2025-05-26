from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import re
import textwrap
import json

def wrap_text_for_markdown_table(text: str, max_width: int = 80) -> str:
    """
    Wrap long text for better presentation in markdown tables.
    Uses <br> line breaks which work in most markdown renderers.

    Args:
        text: The text to wrap
        max_width: Maximum width of a line before wrapping

    Returns:
        Text with <br> line breaks inserted for wrapping
    """
    if not text or len(text) <= max_width:
        return text

    # Split the text into lines
    lines = textwrap.wrap(text, max_width, break_long_words=False, replace_whitespace=False)

    # Join with <br> for markdown line breaks within a cell
    return "<br>".join(lines)

def create_markdown_table(headers: List[str], rows: List[List[str]], wrap_width: int = 80) -> str:
    """
    Create a markdown table with text wrapping for long content and consistent column widths.

    Args:
        headers: List of column headers
        rows: List of rows, each row is a list of cell values
        wrap_width: Maximum width before wrapping text

    Returns:
        Formatted markdown table as a string
    """
    if not headers or not rows:
        return ""

    # Wrap all cell values
    wrapped_rows = []
    for row in rows:
        wrapped_row = [wrap_text_for_markdown_table(str(cell), wrap_width) for cell in row]
        wrapped_rows.append(wrapped_row)

    # Calculate optimal column widths based on content
    col_widths = [len(header) for header in headers]

    # Update column widths based on content in each row
    for row in wrapped_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                # For cells with <br>, measure the longest line
                if '<br>' in cell:
                    lines = cell.split('<br>')
                    max_line_width = max(len(line) for line in lines)
                    col_widths[i] = max(col_widths[i], max_line_width)
                else:
                    col_widths[i] = max(col_widths[i], len(cell))

    # Format the header with proper spacing
    header_row = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"

    # Create the separator row
    separator_row = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    # Format each data row with proper spacing
    formatted_rows = []
    for row in wrapped_rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            if i >= len(col_widths):
                formatted_cells.append(cell)
                continue

            if '<br>' in cell:
                # Handle multiline cells by formatting each line
                lines = cell.split('<br>')
                # Format the first line
                formatted_cell = lines[0].ljust(col_widths[i])
                # Add remaining lines with proper padding
                for line in lines[1:]:
                    formatted_cell += '<br>' + line.ljust(col_widths[i])
                formatted_cells.append(formatted_cell)
            else:
                formatted_cells.append(cell.ljust(col_widths[i]))

        formatted_row = "| " + " | ".join(formatted_cells) + " |"
        formatted_rows.append(formatted_row)

    # Build the complete table
    return "\n".join([header_row, separator_row] + formatted_rows)

def dict_list_to_markdown_table(data: List[Dict[str, Any]], wrap_width: int = 80) -> str:
    """
    Convert a list of dictionaries to a markdown table with wrapped text.

    Args:
        data: List of dictionaries where keys are column names and values are cell values
        wrap_width: Maximum width before wrapping text

    Returns:
        Formatted markdown table as a string
    """
    if not data:
        return ""

    # Extract headers from the first row
    headers = list(data[0].keys())

    # Convert data to rows with proper formatting for complex types
    rows = []
    for item in data:
        row = []
        for header in headers:
            value = item.get(header, "")

            # Handle different value types appropriately
            if value is None:
                value = ""
            # Convert lists to comma-separated strings
            elif isinstance(value, list):
                if len(value) == 0:
                    value = ""
                elif all(isinstance(v, dict) for v in value):
                    # For lists of dictionaries, format each dict and join with semicolons
                    value = "; ".join(["; ".join(f"{k}: {v}" for k, v in d.items()) for d in value])
                else:
                    value = ", ".join(str(v) for v in value)
            # Convert dictionaries to formatted strings
            elif isinstance(value, dict):
                if not value:
                    value = ""
                else:
                    value = "; ".join(f"{k}: {v}" for k, v in value.items())
            # Ensure numeric values are properly formatted
            elif isinstance(value, (int, float)):
                value = str(value)
            # Ensure boolean values are readable
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
            # Convert any other types to strings
            else:
                value = str(value)

            # Clean up the string
            if isinstance(value, str):
                # Remove unnecessary whitespace
                value = value.strip()
                # Escape any pipe characters to prevent table formatting issues
                value = value.replace("|", "\\|")

            row.append(value)
        rows.append(row)

    return create_markdown_table(headers, rows, wrap_width)

def parse_markdown_table(markdown_table: str) -> List[Dict[str, str]]:
    """
    Parse a markdown table into a list of dictionaries.
    Each dictionary represents a row with column names as keys.
    """
    lines = markdown_table.strip().split("\n")

    # Need at least header row, separator row, and one data row
    if len(lines) < 3:
        return []

    # Extract header row and remove leading/trailing pipes
    header = lines[0].strip()
    if header.startswith("|"):
        header = header[1:]
    if header.endswith("|"):
        header = header[:-1]

    # Parse column names
    columns = [col.strip() for col in header.split("|")]

    # Skip separator row (line 1)

    # Process data rows
    result = []
    for i in range(2, len(lines)):
        row = lines[i].strip()
        if not row:
            continue

        # Remove leading/trailing pipes
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]

        # Split row values and handle <br> tags
        values = [val.strip().replace("<br>", "\n") for val in row.split("|")]

        # Create dictionary for row
        if len(values) == len(columns):
            row_dict = {columns[j]: values[j] for j in range(len(columns))}
            result.append(row_dict)

    return result

class LeadQualification(BaseModel):
    """Structured output for Lead Qualification Agent"""
    lead_name: str
    company_name: str
    email_address: str
    phone_number: str
    lead_score: int = Field(..., ge=0, le=100)
    classification: str = Field(..., pattern="^(HOT|WARM|COLD)$")
    reasoning: str
    value_alignment: str
    recommended_approach: str

    @classmethod
    def from_markdown_row(cls, row: Dict[str, str]) -> "LeadQualification":
        """Create a LeadQualification instance from a markdown table row"""
        return cls(
            lead_name=row.get("Lead Name", ""),
            company_name=row.get("Company", ""),
            email_address=row.get("Email", ""),
            phone_number=row.get("Phone", ""),
            lead_score=int(row.get("Lead Score (0-100)", "0")),
            classification=row.get("Classification", "COLD"),
            reasoning=row.get("Reasoning", ""),
            value_alignment=row.get("Value Alignment", ""),
            recommended_approach=row.get("Recommended Approach", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary for markdown table generation"""
        return {
            "Lead Name": self.lead_name,
            "Company": self.company_name,
            "Email": self.email_address,
            "Phone": self.phone_number,
            "Lead Score (0-100)": self.lead_score,
            "Classification": self.classification,
            "Reasoning": self.reasoning,
            "Value Alignment": self.value_alignment,
            "Recommended Approach": self.recommended_approach
        }

class CompanyProfile(BaseModel):
    """Company information for prospect research"""
    industry: str
    size: str
    revenue: Optional[str] = Field(default="Information not readily available")
    funding: Optional[str] = Field(default="Information not readily available")
    growth_trajectory: str
    tech_stack: Optional[str] = Field(default="Information not readily available")
    products_services: str
    organizational_structure: Optional[str] = Field(default="Information not readily available")
    corporate_initiatives: Optional[str] = Field(default="Information not readily available")
    market_position: Optional[str] = Field(default="Information not readily available")
    competitive_landscape: Optional[str] = Field(default="Information not readily available")

    def to_string(self) -> str:
        """Format company profile for display in markdown table"""
        parts = [
            f"Industry: {self.industry}",
            f"Size: {self.size}",
            f"Revenue: {self.revenue}",
            f"Funding: {self.funding}",
            f"Growth: {self.growth_trajectory}",
            f"Tech Stack: {self.tech_stack}",
            f"Products/Services: {self.products_services}"
        ]

        # Add optional fields if they contain meaningful information
        if self.organizational_structure and self.organizational_structure != "Information not readily available":
            parts.append(f"Organization: {self.organizational_structure}")
        if self.corporate_initiatives and self.corporate_initiatives != "Information not readily available":
            parts.append(f"Initiatives: {self.corporate_initiatives}")
        if self.market_position and self.market_position != "Information not readily available":
            parts.append(f"Market Position: {self.market_position}")
        if self.competitive_landscape and self.competitive_landscape != "Information not readily available":
            parts.append(f"Competitive Landscape: {self.competitive_landscape}")

        return "<br>".join(parts)

class DecisionMakerProfile(BaseModel):
    """Decision maker information for prospect research"""
    role: str
    responsibilities: str
    career_history: Optional[str] = Field(default="Information not readily available")
    educational_background: Optional[str] = Field(default="Information not readily available")
    social_media_presence: Optional[str] = Field(default="Information not readily available")
    expertise_areas: Optional[str] = Field(default="Information not readily available")
    professional_achievements: Optional[str] = Field(default="Information not readily available")
    publications_presentations: Optional[str] = Field(default="Information not readily available")
    decision_making_style: Optional[str] = Field(default="Information not readily available")
    network_relationships: Optional[str] = Field(default="Information not readily available")

    def to_string(self) -> str:
        """Format decision maker profile for display in markdown table"""
        parts = [
            f"Role: {self.role}",
            f"Responsibilities: {self.responsibilities}"
        ]

        # Add career history if available
        if self.career_history and self.career_history != "Information not readily available":
            parts.append(f"Career History: {self.career_history}")

        # Add educational background if available
        if self.educational_background and self.educational_background != "Information not readily available":
            parts.append(f"Education: {self.educational_background}")

        # Add social media presence if available
        if self.social_media_presence and self.social_media_presence != "Information not readily available":
            parts.append(f"Social Media: {self.social_media_presence}")

        # Add expertise areas if available
        if self.expertise_areas and self.expertise_areas != "Information not readily available":
            parts.append(f"Expertise: {self.expertise_areas}")

        # Add achievements if available
        if self.professional_achievements and self.professional_achievements != "Information not readily available":
            parts.append(f"Achievements: {self.professional_achievements}")

        # Add publications if available
        if self.publications_presentations and self.publications_presentations != "Information not readily available":
            parts.append(f"Publications: {self.publications_presentations}")

        # Add decision-making style if available
        if self.decision_making_style and self.decision_making_style != "Information not readily available":
            parts.append(f"Decision Style: {self.decision_making_style}")

        # Add network relationships if available
        if self.network_relationships and self.network_relationships != "Information not readily available":
            parts.append(f"Network: {self.network_relationships}")

        return "<br>".join(parts)

class ProspectResearch(BaseModel):
    """Structured output for Prospect Research Agent"""
    lead_name: str
    company_name: str
    classification: str = Field(..., pattern="^(HOT|WARM)$")
    company_profile: CompanyProfile
    decision_maker_profile: DecisionMakerProfile
    recent_news: List[str]
    pain_points: List[str]
    current_solutions: Optional[str] = Field(default="Information not readily available")
    urgency_evidence: str
    recommended_talking_points: List[str]
    sources_cited: Optional[List[str]] = Field(default_factory=list)

    @classmethod
    def from_markdown_row(cls, row: Dict[str, str]) -> "ProspectResearch":
        """Create a ProspectResearch instance from a markdown table row"""
        # Parse company profile data from detailed text
        company_profile_text = row.get("Company Profile", "")

        # Extract component parts using pattern matching or defaults
        company_profile = CompanyProfile(
            industry=_extract_field(company_profile_text, "Industry", default="Unknown industry"),
            size=_extract_field(company_profile_text, "Size", default="Unknown size"),
            revenue=_extract_field(company_profile_text, "Revenue"),
            funding=_extract_field(company_profile_text, "Funding"),
            growth_trajectory=_extract_field(company_profile_text, "Growth", default="Unknown growth"),
            tech_stack=_extract_field(company_profile_text, "Tech Stack"),
            products_services=_extract_field(company_profile_text, "Products/Services", default="Unknown products/services"),
            organizational_structure=_extract_field(company_profile_text, "Organization"),
            corporate_initiatives=_extract_field(company_profile_text, "Initiatives"),
            market_position=_extract_field(company_profile_text, "Market Position"),
            competitive_landscape=_extract_field(company_profile_text, "Competitive Landscape")
        )

        # Parse decision maker profile data
        role_background_text = row.get("Role & Background", "")

        decision_maker_profile = DecisionMakerProfile(
            role=_extract_field(role_background_text, "Role", default="Unknown role"),
            responsibilities=_extract_field(role_background_text, "Responsibilities", default="Unknown responsibilities"),
            career_history=_extract_field(role_background_text, "Career History"),
            educational_background=_extract_field(role_background_text, "Education"),
            social_media_presence=_extract_field(role_background_text, "Social Media"),
            expertise_areas=_extract_field(role_background_text, "Expertise"),
            professional_achievements=_extract_field(role_background_text, "Achievements"),
            publications_presentations=_extract_field(role_background_text, "Publications"),
            decision_making_style=_extract_field(role_background_text, "Decision Style"),
            network_relationships=_extract_field(role_background_text, "Network")
        )

        # Parse recent developments as a list, handling different formats
        recent_news_text = row.get("Recent Developments", "")
        if "<br>" in recent_news_text:
            recent_news = [news.strip() for news in recent_news_text.split("<br>") if news.strip()]
        else:
            recent_news = [news.strip() for news in recent_news_text.split(",") if news.strip()]

        if not recent_news:
            recent_news = ["Information not readily available"]

        # Parse pain points, handling numbered lists
        pain_points_text = row.get("Pain Points", "")
        pain_points = []

        # Handle numbered lists like "1. Point one<br>2. Point two"
        if re.search(r'\d+\.', pain_points_text):
            for point in re.split(r'<br>|\n', pain_points_text):
                clean_point = re.sub(r'^\d+\.\s*', '', point.strip())
                if clean_point:
                    pain_points.append(clean_point)
        # Handle simple comma-separated or <br>-separated lists
        elif "<br>" in pain_points_text:
            pain_points = [p.strip() for p in pain_points_text.split("<br>") if p.strip()]
        else:
            pain_points = [p.strip() for p in pain_points_text.split(",") if p.strip()]

        if not pain_points:
            pain_points = ["Information not readily available"]

        # Parse talking points, handling different formats
        talking_points_text = row.get("Recommended Talking Points", "")
        talking_points = []

        # Handle numbered lists
        if re.search(r'\d+\.', talking_points_text):
            for point in re.split(r'<br>|\n', talking_points_text):
                clean_point = re.sub(r'^\d+\.\s*', '', point.strip())
                if clean_point:
                    talking_points.append(clean_point)
        # Handle simple comma-separated or <br>-separated lists
        elif "<br>" in talking_points_text:
            talking_points = [p.strip() for p in talking_points_text.split("<br>") if p.strip()]
        else:
            talking_points = [p.strip() for p in talking_points_text.split(",") if p.strip()]

        if not talking_points:
            talking_points = ["Information not readily available"]

        # Extract source citations if present in parentheses or after "according to"
        sources = []
        for field in [company_profile_text, role_background_text, recent_news_text, pain_points_text,
                     row.get("Current Solutions", ""), row.get("Urgency Factors", ""), talking_points_text]:
            # Find all text in parentheses that looks like a source
            parenthetical_sources = re.findall(r'\((?:per|from|via|according to|mentioned in|cited in|from)\s+([^)]+)\)', field)
            sources.extend(parenthetical_sources)

            # Find all "according to" phrases
            according_to_sources = re.findall(r'according to\s+([^,\.;]+)', field, re.IGNORECASE)
            sources.extend(according_to_sources)

            # Find all "mentioned in" phrases
            mentioned_in_sources = re.findall(r'mentioned in\s+([^,\.;]+)', field, re.IGNORECASE)
            sources.extend(mentioned_in_sources)

        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)

        return cls(
            lead_name=row.get("Lead Name", ""),
            company_name=row.get("Company", ""),
            classification=row.get("Classification", "HOT"),
            company_profile=company_profile,
            decision_maker_profile=decision_maker_profile,
            recent_news=recent_news,
            pain_points=pain_points,
            current_solutions=row.get("Current Solutions", "Information not readily available"),
            urgency_evidence=row.get("Urgency Factors", "Information not readily available"),
            recommended_talking_points=talking_points,
            sources_cited=unique_sources
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary for markdown table generation"""
        # Format the pain points as a numbered list for better readability
        formatted_pain_points = []
        for i, point in enumerate(self.pain_points):
            formatted_pain_points.append(f"{i+1}. {point}")

        # Format talking points as a numbered list
        formatted_talking_points = []
        for i, point in enumerate(self.recommended_talking_points):
            formatted_talking_points.append(f"{i+1}. {point}")

        # Format recent news as separate items
        formatted_news = []
        for news in self.recent_news:
            formatted_news.append(news)

        return {
            "Lead Name": self.lead_name,
            "Company": self.company_name,
            "Classification": self.classification,
            "Company Profile": self.company_profile.to_string(),
            "Role & Background": self.decision_maker_profile.to_string(),
            "Recent Developments": "<br>".join(formatted_news),
            "Pain Points": "<br>".join(formatted_pain_points),
            "Current Solutions": self.current_solutions,
            "Urgency Factors": self.urgency_evidence,
            "Recommended Talking Points": "<br>".join(formatted_talking_points)
        }

def _extract_field(text: str, field_name: str, default: str = "Information not readily available") -> str:
    """Extract a field from a formatted string using various patterns"""
    # Try to find patterns like "Field: value" or "Field - value"
    patterns = [
        rf"{field_name}:\s*([^,<\n]+)",
        rf"{field_name}\s*-\s*([^,<\n]+)",
        rf"{field_name}\s+([^,<\n]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # For fields that might span multiple lines or contain more complex content
    if "<br>" in text:
        lines = text.split("<br>")
        for line in lines:
            if field_name.lower() in line.lower():
                # Return everything after the field name and any delimiter
                value = re.sub(rf"^.*{field_name}[:\s-]*", "", line, flags=re.IGNORECASE)
                if value.strip():
                    return value.strip()

    return default

class EmailTemplate(BaseModel):
    """Structured output for Email Outreach Agent"""
    lead_name: str
    company_name: str
    classification: str = Field(..., pattern="^(HOT|WARM)$")
    subject_line: str
    email_body: str
    follow_up_timing: str
    alternative_contact_channels: Optional[str] = Field(default=None)
    ab_test_variations: List[dict]

    @classmethod
    def from_markdown_row(cls, row: Dict[str, str]) -> "EmailTemplate":
        """Create an EmailTemplate instance from a markdown table row"""
        # Parse AB test variations
        ab_test_str = row.get("A/B Test Variations", "")
        ab_test_variations = []

        if ab_test_str:
            variations = [v.strip() for v in ab_test_str.split(";")]
            for variation in variations:
                if ":" in variation:
                    key, value = variation.split(":", 1)
                    ab_test_variations.append({"type": key.strip(), "content": value.strip()})

        if not ab_test_variations:
            ab_test_variations = [{"type": "No variations", "content": ""}]

        return cls(
            lead_name=row.get("Lead Name", ""),
            company_name=row.get("Company", ""),
            classification=row.get("Classification", "WARM"),
            subject_line=row.get("Subject Line", ""),
            email_body=row.get("Email Body", ""),
            follow_up_timing=row.get("Follow-up Timing", "3 days"),
            alternative_contact_channels=row.get("Alternative Contact", None),
            ab_test_variations=ab_test_variations
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary for markdown table generation"""
        # Format email body with proper line breaks for readability in markdown
        formatted_email_body = self.email_body.replace("\n\n", "<br><br>").replace("\n", "<br>")

        # Format A/B test variations in a readable way
        ab_test_str = "; ".join([f"{v['type']}: {v['content']}" for v in self.ab_test_variations if v['content']])
        if not ab_test_str:
            ab_test_str = "None"

        return {
            "Lead Name": self.lead_name,
            "Company": self.company_name,
            "Classification": self.classification,
            "Subject Line": self.subject_line,
            "Email Body": formatted_email_body,
            "Follow-up Timing": self.follow_up_timing,
            "Alternative Contact": self.alternative_contact_channels or "None",
            "A/B Test Variations": ab_test_str
        }

class CallBrief(BaseModel):
    """Structured output for Sales Call Preparation Agent"""
    lead_name: str
    company_name: str
    classification: str = Field(..., pattern="^(HOT|WARM)$")
    company_snapshot: str
    decision_maker_profile: str
    relationship_history: Optional[str] = Field(default="No previous interactions")
    pain_points: List[str]
    talking_points: List[str]
    objection_responses: List[Dict[str, str]] = Field(default_factory=list)
    next_steps: List[str]
    recent_developments: str
    competitive_insights: Optional[str] = Field(default=None)
    value_propositions: List[str]

    @classmethod
    def from_markdown_row(cls, row: Dict[str, str]) -> "CallBrief":
        """Create a CallBrief instance from a markdown table row"""
        # Parse pain points and talking points as lists
        pain_points = [point.strip() for point in row.get("Pain Points", "").split(",") if point.strip()]
        if not pain_points:
            pain_points = ["No specific pain points identified"]

        talking_points = [point.strip() for point in row.get("Talking Points", "").split(",") if point.strip()]
        if not talking_points:
            talking_points = ["General introduction to our solution"]

        # Parse objection responses
        objection_str = row.get("Objection Responses", "")
        objection_responses = []

        if objection_str:
            objections = [obj.strip() for obj in objection_str.split(";")]
            for objection in objections:
                if ":" in objection:
                    key, value = objection.split(":", 1)
                    objection_responses.append({"objection": key.strip(), "response": value.strip()})

        # Parse next steps as a list
        next_steps = [step.strip() for step in row.get("Next Steps", "").split(",") if step.strip()]
        if not next_steps:
            next_steps = ["Schedule follow-up call"]

        # Parse value propositions as a list
        value_props = [prop.strip() for prop in row.get("Value Proposition", "").split(",") if prop.strip()]
        if not value_props:
            value_props = ["Our solution addresses key pain points"]

        return cls(
            lead_name=row.get("Lead Name", ""),
            company_name=row.get("Company", ""),
            classification=row.get("Classification", "WARM"),
            company_snapshot=row.get("Company Snapshot", ""),
            decision_maker_profile=row.get("Decision-maker Profile", ""),
            relationship_history=row.get("Relationship History", "No previous interactions"),
            pain_points=pain_points,
            talking_points=talking_points,
            objection_responses=objection_responses,
            next_steps=next_steps,
            recent_developments=row.get("Recent Developments", ""),
            competitive_insights=row.get("Competitive Insights", None),
            value_propositions=value_props
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary for markdown table generation"""
        # Format lists with numeric prefixes for better readability
        def format_list_items(items):
            if not items:
                return "None"
            return "<br>".join([f"{i+1}. {item}" for i, item in enumerate(items)])

        # Format objection responses in a readable format
        objection_str = ""
        if self.objection_responses:
            objection_items = []
            for i, obj in enumerate(self.objection_responses):
                if 'objection' in obj and 'response' in obj:
                    objection_items.append(f"{i+1}. {obj['objection']}: {obj['response']}")
            objection_str = "<br>".join(objection_items)

        if not objection_str:
            objection_str = "None anticipated"

        return {
            "Lead Name": self.lead_name,
            "Company": self.company_name,
            "Classification": self.classification,
            "Company Snapshot": self.company_snapshot,
            "Decision-maker Profile": self.decision_maker_profile,
            "Relationship History": self.relationship_history or "No previous interactions",
            "Pain Points": format_list_items(self.pain_points),
            "Talking Points": format_list_items(self.talking_points),
            "Objection Responses": objection_str,
            "Next Steps": format_list_items(self.next_steps),
            "Recent Developments": self.recent_developments or "None recent",
            "Competitive Insights": self.competitive_insights or "None available",
            "Value Proposition": format_list_items(self.value_propositions)
        }

class LeadQualificationResponse(BaseModel):
    """Container for multiple lead qualification records"""
    qualified_leads: List[LeadQualification]

    @classmethod
    def from_markdown_table(cls, markdown_table: str) -> "LeadQualificationResponse":
        """Create a LeadQualificationResponse from a markdown table"""
        rows = parse_markdown_table(markdown_table)
        return cls(qualified_leads=[LeadQualification.from_markdown_row(row) for row in rows])

    def to_markdown_table(self, wrap_width: int = 80) -> str:
        """Convert to a markdown table with text wrapping for long content"""
        if not self.qualified_leads:
            return "No lead qualification records available."

        # Convert all lead records to dictionaries
        lead_dicts = [lead.to_dict() for lead in self.qualified_leads]

        # Generate markdown table with wrapped text
        return dict_list_to_markdown_table(lead_dicts, wrap_width)

class ProspectResearchResponse(BaseModel):
    """Container for multiple prospect research records"""
    enriched_leads: List[ProspectResearch]

    @classmethod
    def from_markdown_table(cls, markdown_table: str) -> "ProspectResearchResponse":
        """Create a ProspectResearchResponse from a markdown table"""
        rows = parse_markdown_table(markdown_table)
        return cls(enriched_leads=[ProspectResearch.from_markdown_row(row) for row in rows])

    def to_markdown_table(self, wrap_width: int = 80) -> str:
        """Convert to a markdown table with text wrapping for long content"""
        if not self.enriched_leads:
            return "No prospect research records available."

        # Convert all research records to dictionaries
        lead_dicts = [lead.to_dict() for lead in self.enriched_leads]

        # Generate markdown table with wrapped text
        return dict_list_to_markdown_table(lead_dicts, wrap_width)

class EmailOutreachResponse(BaseModel):
    """Container for multiple email templates"""
    email_templates: List[EmailTemplate] = Field(default_factory=list)

    @classmethod
    def from_markdown_table(cls, markdown_table: str) -> "EmailOutreachResponse":
        """Create a EmailOutreachResponse from a markdown table"""
        rows = parse_markdown_table(markdown_table)
        return cls(email_templates=[EmailTemplate.from_markdown_row(row) for row in rows])

    def to_markdown_table(self, wrap_width: int = 80) -> str:
        """Convert to a markdown representation"""
        if not self.email_templates:
            return "# Email Outreach Error\n\nThe requested Markdown document cannot be generated at this time due to incomplete lead data.  Please provide complete and accurate lead data including Lead Name, Company, Classification, Subject Line, Email Body, Follow-up Timing, Alternative Contact Channels, and A/B Test Variations in JSON format.  The task will be completed upon receipt of the complete data."

        # Convert to dictionary for jsonschema2md
        data = {"email_templates": [template.model_dump() for template in self.email_templates]}
        return json.dumps(data)

class SalesCallPrepResponse(BaseModel):
    """Container for multiple call briefs"""
    call_briefs: List[CallBrief] = Field(default_factory=list)

    @classmethod
    def from_markdown_table(cls, markdown_table: str) -> "SalesCallPrepResponse":
        """Create a SalesCallPrepResponse from a markdown table"""
        rows = parse_markdown_table(markdown_table)
        return cls(call_briefs=[CallBrief.from_markdown_row(row) for row in rows])

    def to_markdown_table(self, wrap_width: int = 80) -> str:
        """Convert to a markdown representation"""
        if not self.call_briefs:
            return "# Sales Call Prep Error\n\nThe task cannot be completed due to incomplete data. The provided lead data lacks the necessary details for creating personalized call briefs. To generate the required Markdown document, the Sales Call Preparation Agent must provide fully enriched data for each lead. The JSON data should be an array of objects, where each object represents a call brief and includes the following fields: `Lead Name`, `Company`, `Classification`, `Company Snapshot`, `Decision-maker Profile`, `Relationship History`, `Pain Points`, `Talking Points`, `Objection Responses`, `Next Steps`, `Recent Developments`, `Competitive Insights`, and `Value Propositions`. Only when this complete data is provided can the conversion to a Markdown document proceed."

        # Convert to dictionary for jsonschema2md
        data = {"call_briefs": [brief.model_dump() for brief in self.call_briefs]}
        return json.dumps(data)
