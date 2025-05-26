#!/usr/bin/env python

import json
import sys
import os
from pathlib import Path
import jsonschema2md

# Add the parent directory to the path so we can import from sales_guru
sys.path.append(str(Path(__file__).parent.parent.parent))

from sales_guru.schemas import SalesCallPrepResponse, CallBrief, EmailOutreachResponse, EmailTemplate
from sales_guru.tools.json_to_markdown import JSONToMarkdownTool


def read_sample_data():
    """Read the sample data from the sales_call_prep.md file."""
    sample_data_path = Path(__file__).parent / "outputs" / "sales_call_prep.md"
    
    print(f"Reading sample data from {sample_data_path}")
    
    with open(sample_data_path, "r") as f:
        content = f.read()
    
    return content


def create_sample_call_brief():
    """Create a sample call brief object."""
    brief = CallBrief(
        lead_name="John Doe",
        company_name="Sample Corp",
        classification="HOT",
        company_snapshot="A sample company for testing.",
        decision_maker_profile="CEO with decision-making authority.",
        relationship_history="Initial contact via email.",
        pain_points=["High costs", "Inefficient processes"],
        talking_points=["Our solution can reduce costs", "We offer streamlined workflows"],
        objection_responses=[
            {"objection": "Too expensive", "response": "ROI within 6 months"}
        ],
        next_steps=["Schedule demo", "Send proposal"],
        recent_developments="Recently expanded operations.",
        competitive_insights="Competing with XYZ Corp.",
        value_propositions=["Cost savings", "Efficiency gains"]
    )
    
    return brief


def create_sample_email_template():
    """Create a sample email template object."""
    template = EmailTemplate(
        lead_name="Andy Jassy",
        company_name="Amazon",
        classification="HOT",
        subject_line="Revolutionize Amazon's Logistics with AI-Powered Automation",
        email_body="""Dear Andy,

I hope this email finds you well. I recently read about Amazon's increased investment in AI and AWS in the Wall Street Journal, and was particularly impressed by your emphasis on AI as a key strategic initiative.

At Oceaneering, we've developed AI-integrated solutions that can significantly enhance Amazon's logistics network efficiency while reducing operational costs. Our autonomous material handling systems have helped companies similar to Amazon achieve 30% increases in throughput and 25% reductions in labor costs.

I'd appreciate the opportunity to discuss how our solutions could complement Amazon's existing technology stack. Would you be available for a 30-minute call next Tuesday at 10:00 AM PT?

Looking forward to your response.

Best regards,
Sarah Johnson
VP of Business Development
Oceaneering
sarah.johnson@oceaneering.com
(555) 123-4567""",
        follow_up_timing="3 days if no response",
        alternative_contact_channels="LinkedIn InMail to Andy Jassy",
        ab_test_variations=[
            {"type": "Subject Line", "content": "Boost Amazon's Efficiency with AI-Powered Logistics Solutions"},
            {"type": "Opening", "content": "Your recent interview with CNBC about Amazon's focus on AI innovation was insightful."}
        ]
    )
    
    return template


def main():
    """Test the markdown conversion with both approaches."""
    
    # Create a sample SalesCallPrepResponse
    print("Creating sample data...")
    sample_brief = create_sample_call_brief()
    response = SalesCallPrepResponse(call_briefs=[sample_brief])
    
    # Convert to JSON using the to_markdown_table method
    print("\nConverting call brief to JSON...")
    json_output = response.to_markdown_table()
    print(f"JSON output: {json_output[:100]}...")
    
    # Convert JSON to markdown using the JSONToMarkdownTool
    print("\nConverting JSON to Markdown...")
    tool = JSONToMarkdownTool()
    markdown_output = tool._run(json_output, title="Sales Call Brief Test")
    
    # Save the markdown output to a file
    output_path = Path(__file__).parent / "outputs" / "test_markdown_output.md"
    
    print(f"\nSaving markdown output to {output_path}")
    with open(output_path, "w") as f:
        f.write(markdown_output)
    
    print("\nMarkdown output:")
    print("-" * 50)
    print(markdown_output[:500] + "..." if len(markdown_output) > 500 else markdown_output)
    print("-" * 50)
    
    # Test EmailOutreachResponse conversion
    print("\nTesting email outreach conversion...")
    sample_email = create_sample_email_template()
    email_response = EmailOutreachResponse(email_templates=[sample_email])
    
    # Convert to JSON
    print("\nConverting email template to JSON...")
    email_json = email_response.to_markdown_table()
    print(f"Email JSON output: {email_json[:100]}...")
    
    # Convert JSON to markdown
    print("\nConverting email JSON to Markdown...")
    email_md_output = tool._run(email_json, title="Email Outreach Test")
    
    # Save the email markdown output
    email_output_path = Path(__file__).parent / "outputs" / "email_markdown_output.md"
    
    print(f"\nSaving email markdown output to {email_output_path}")
    with open(email_output_path, "w") as f:
        f.write(email_md_output)
    
    print("\nEmail markdown output preview:")
    print("-" * 50)
    print(email_md_output[:500] + "..." if len(email_md_output) > 500 else email_md_output)
    print("-" * 50)
    
    # Now let's try to process the existing sales_call_prep.md file
    print("\nProcessing existing sales_call_prep.md file...")
    
    # Read the sample data
    original_md = read_sample_data()
    
    # Parse the markdown table into JSON data
    # For simplicity, we'll create a mock structure rather than parsing the table
    # Create a more complex example from the sales_call_prep.md content
    sample_data = {
        "call_briefs": [
            {
                "lead_name": "Andy Jassy",
                "company_name": "Amazon",
                "classification": "HOT",
                "company_snapshot": "Multinational e-commerce and cloud computing giant. Publicly traded (NASDAQ: AMZN). Strong focus on AI and increased capital expenditures in AWS and generative AI.",
                "decision_maker_profile": "Andy Jassy, CEO. Extensive experience in technology and business strategy. Previously led Amazon Web Services (AWS) to significant growth.",
                "relationship_history": "Initial contact via email. Responded positively to our outreach highlighting AI-powered efficiency solutions.",
                "pain_points": [
                    "Maintaining market leadership in a highly competitive landscape",
                    "Balancing rapid innovation with operational efficiency and profitability",
                    "Managing a large and complex global workforce"
                ],
                "talking_points": [
                    "Highlight Oceaneering's scalable and efficient solutions for Amazon's logistics network.",
                    "Showcase how our AI-integrated solutions can enhance Amazon's customer experience and operational efficiency.",
                    "Position our offerings as complementary to Amazon's existing technology stack and strategic initiatives."
                ],
                "objection_responses": [
                    {"objection": "Too expensive", "response": "Our solutions offer significant long-term ROI through increased efficiency and reduced labor costs. We can tailor a solution to fit your budget and scale."},
                    {"objection": "Integration challenges", "response": "Our team has extensive experience integrating with large-scale systems. We'll work closely with your IT team to ensure a smooth transition."}
                ],
                "next_steps": [
                    "Schedule a meeting to discuss Amazon's specific needs.",
                    "Provide a customized proposal outlining potential cost savings and efficiency gains.",
                    "Arrange a demonstration of our autonomous material handling solutions."
                ],
                "recent_developments": "Amazon's increased investment in AI and AWS; Jassy's emphasis on AI as a key strategic initiative.",
                "competitive_insights": "Competitors lack the scalability and AI integration capabilities of Oceaneering's solutions.",
                "value_propositions": [
                    "Increased efficiency and productivity", 
                    "Reduced labor costs", 
                    "Improved safety and reliability", 
                    "Scalable solutions to meet Amazon's growing needs"
                ]
            },
            {
                "lead_name": "Doug McMillon",
                "company_name": "Walmart Inc.",
                "classification": "HOT",
                "company_snapshot": "World's largest retailer by revenue. Publicly traded. Significant investments in supply chain technology and automation.",
                "decision_maker_profile": "Doug McMillon, President and CEO. Extensive experience in retail, supply chain management, and logistics.",
                "relationship_history": "Initial contact via email. Walmart has shown interest in supply chain technology and automation.",
                "pain_points": [
                    "Maintaining efficiency and cost-effectiveness in a large-scale distribution network",
                    "Meeting customer demands for faster delivery and wider product selection",
                    "Managing labor costs and maintaining employee morale"
                ],
                "talking_points": [
                    "Highlight Oceaneering's cost savings and efficiency improvements.",
                    "Emphasize our ability to integrate with Walmart's existing systems.",
                    "Showcase success stories with similar large-scale retailers."
                ],
                "objection_responses": [
                    {"objection": "Too risky", "response": "Our phased rollout minimizes disruption and risk, ensuring a smooth transition to automated systems."},
                    {"objection": "Too expensive", "response": "Our ROI analysis demonstrates significant cost savings over time through reduced labor and improved efficiency."}
                ],
                "next_steps": [
                    "Schedule a meeting to discuss Walmart's specific needs.",
                    "Provide a customized proposal outlining potential cost savings and efficiency gains.",
                    "Arrange a demonstration of our autonomous material handling solutions."
                ],
                "recent_developments": "Walmart's investments in supply chain technology and automation.",
                "competitive_insights": "Our solutions offer superior scalability and integration capabilities compared to competitors.",
                "value_propositions": [
                    "Significant cost savings",
                    "Increased efficiency and productivity",
                    "Improved order fulfillment speed",
                    "Enhanced employee safety"
                ]
            }
        ]
    }
    
    # Convert the sample data to JSON
    complex_json = json.dumps(sample_data)
    
    # Use our tool to convert it to markdown
    complex_md_output = tool._run(complex_json, title="Sales Call Preparation")
    
    # Save the complex markdown output
    complex_output_path = Path(__file__).parent / "outputs" / "complex_markdown_output.md"
    
    print(f"\nSaving complex markdown output to {complex_output_path}")
    with open(complex_output_path, "w") as f:
        f.write(complex_md_output)
    
    print("\nComplex markdown conversion completed!")
    print(f"\nTest complete. Full output written to {output_path} and {complex_output_path}")


if __name__ == "__main__":
    main() 