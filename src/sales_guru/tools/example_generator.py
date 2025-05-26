import json
from typing import Any
from crewai.tools import BaseTool

class ExampleGeneratorTool(BaseTool):
    """Tool for generating example output structures for agents to follow."""

    name: str = "ExampleGenerator"
    description: str = """
    Use this tool to get example output structures for different agent tasks.
    Input should be the name of the task you need an example for:
    - 'lead_qualification' - Get example of valid lead qualification output
    - 'prospect_research' - Get example of valid prospect research output
    - 'email_outreach' - Get example of valid email outreach output
    - 'sales_call_prep' - Get example of valid sales call preparation output

    The tool will return a JSON example structure that you can use as a template.
    """

    def _run(self, task_name: str) -> str:
        """Generate an example output structure for the specified task."""
        try:
            task_name = task_name.lower().strip()

            if task_name == 'lead_qualification':
                return self._generate_lead_qualification_example()
            elif task_name == 'prospect_research':
                return self._generate_prospect_research_example()
            elif task_name == 'email_outreach':
                return self._generate_email_outreach_example()
            elif task_name == 'sales_call_prep':
                return self._generate_sales_call_prep_example()
            else:
                return f"Error: Unknown task '{task_name}'. Valid tasks are: lead_qualification, prospect_research, email_outreach, sales_call_prep"

        except Exception as e:
            return f"Error generating example: {str(e)}"

    def _generate_lead_qualification_example(self) -> str:
        """Generate example lead qualification output."""
        example = {
            "qualified_leads": [
                {
                    "lead_name": "John Smith",
                    "company_name": "ABC Corporation",
                    "email_address": "john.smith@abccorp.com",
                    "phone_number": "555-123-4567",
                    "lead_score": 85,
                    "classification": "HOT",
                    "reasoning": "Decision maker at a company with immediate need for our solution.",
                    "value_alignment": "Their automation needs align perfectly with our product capabilities.",
                    "recommended_approach": "Direct outreach highlighting ROI and implementation timeline."
                },
                {
                    "lead_name": "Jane Doe",
                    "company_name": "XYZ Industries",
                    "email_address": "jane.doe@xyzind.com",
                    "phone_number": "555-987-6543",
                    "lead_score": 65,
                    "classification": "WARM",
                    "reasoning": "Expressed interest but timeline is uncertain.",
                    "value_alignment": "Their growth plans align with our solution's scalability.",
                    "recommended_approach": "Educational content focusing on case studies."
                }
            ]
        }
        return json.dumps(example, indent=2)

    def _generate_prospect_research_example(self) -> str:
        """Generate example prospect research output."""
        example = {
            "enriched_leads": [
                {
                    "lead_name": "John Smith",
                    "company_name": "ABC Corporation",
                    "classification": "HOT",
                    "company_profile": {
                        "industry": "Manufacturing",
                        "size": "500-1000 employees",
                        "revenue": "$50M-$100M annually",
                        "funding": "Series C, $25M in 2022",
                        "growth_trajectory": "Expanding operations in APAC region",
                        "tech_stack": "SAP, Microsoft Azure, custom ERP",
                        "products_services": "Industrial automation equipment",
                        "organizational_structure": "Decentralized with regional autonomy",
                        "corporate_initiatives": "Digital transformation, sustainability",
                        "market_position": "Market leader in industrial sensors",
                        "competitive_landscape": "Facing new competition from startups"
                    },
                    "decision_maker_profile": {
                        "role": "VP of Operations",
                        "responsibilities": "Manufacturing efficiency and automation",
                        "career_history": "15+ years in manufacturing tech",
                        "educational_background": "MBA, Mechanical Engineering degree",
                        "social_media_presence": "Active on LinkedIn, industry speaker",
                        "expertise_areas": "Process optimization, automation",
                        "professional_achievements": "Reduced production costs by 30%",
                        "publications_presentations": "Speaker at Manufacturing Tech Conference",
                        "decision_making_style": "Data-driven, consensus builder",
                        "network_relationships": "Well-connected with industry leaders"
                    },
                    "recent_news": [
                        "Announced expansion to new facility",
                        "Released quarterly earnings (15% growth)",
                        "Launched new product line"
                    ],
                    "pain_points": [
                        "Legacy systems causing inefficiencies",
                        "High labor costs in material handling",
                        "Competitor pressure to modernize operations"
                    ],
                    "current_solutions": "Using manual processes with outdated equipment",
                    "urgency_evidence": "CEO mentioned automation priority in recent earnings call",
                    "recommended_talking_points": [
                        "ROI analysis showing 40% cost reduction",
                        "Implementation timeline under 3 months",
                        "Compatibility with existing SAP systems",
                        "Case study of similar manufacturer"
                    ],
                    "sources_cited": [
                        "Company website",
                        "LinkedIn profile",
                        "Q2 2023 Earnings Report",
                        "Industry analysis from Gartner"
                    ]
                }
            ]
        }
        return json.dumps(example, indent=2)

    def _generate_email_outreach_example(self) -> str:
        """Generate example email outreach output."""
        example = {
            "email_templates": [
                {
                    "lead_name": "John Smith",
                    "company_name": "ABC Corporation",
                    "classification": "HOT",
                    "subject_line": "Reducing Material Handling Costs at ABC Corporation by 40%",
                    "email_body": "Dear John,\n\nI noticed ABC Corporation's recent expansion announcement and your focus on operational efficiency mentioned in your Q2 earnings call.\n\nOur autonomous material handling solutions have helped similar manufacturers reduce labor costs by 40% while increasing throughput by 25%. Given your current initiatives in automation, I thought you might be interested in how our technology integrates with your existing SAP systems.\n\nWould you be open to a brief call next week to discuss how we've helped companies like XYZ Industries modernize their operations in under 3 months?\n\nBest regards,\nSales Representative",
                    "follow_up_timing": "3 days",
                    "alternative_contact_channels": "LinkedIn InMail, Connect via industry conference next month",
                    "ab_test_variations": [
                        {
                            "element": "subject_line",
                            "variation": "John, how ABC Corporation can reduce warehouse costs by 40%"
                        },
                        {
                            "element": "opening",
                            "variation": "Congratulations on ABC's expansion plans announced last week. As you scale operations, I wanted to share how our autonomous material handling solutions might align with your automation initiatives."
                        }
                    ]
                }
            ]
        }
        return json.dumps(example, indent=2)

    def _generate_sales_call_prep_example(self) -> str:
        """Generate example sales call preparation output."""
        example = {
            "call_briefs": [
                {
                    "lead_name": "John Smith",
                    "company_name": "ABC Corporation",
                    "classification": "HOT",
                    "company_snapshot": "Manufacturing company with 750 employees, $75M revenue, growing at 15% annually. Recently expanded operations with new facility in APAC region. Industry leader in industrial sensors with increasing competitive pressure.",
                    "decision_maker_profile": "John Smith, VP of Operations, 15+ years in manufacturing tech. MBA, mechanical engineering background. Data-driven decision maker who values ROI and implementation feasibility. Active in industry events and well-connected.",
                    "relationship_history": "Initial contact via email 2 weeks ago. Responded positively to ROI calculator we sent. Downloaded automation guide from website.",
                    "pain_points": [
                        "High labor costs in material handling operations",
                        "Inefficient legacy systems slowing down production",
                        "Pressure from board to modernize operations",
                        "Difficulty finding qualified warehouse staff"
                    ],
                    "talking_points": [
                        "Our autonomous forklift solution reduces labor costs by 40% with ROI in 18 months",
                        "Integration with existing SAP systems takes under 2 weeks",
                        "Similar manufacturer XYZ Industries achieved 25% throughput increase",
                        "Flexible deployment options (purchase, lease, robotics-as-a-service)"
                    ],
                    "objection_responses": [
                        {
                            "objection": "Too expensive compared to human labor",
                            "response": "While the initial investment is higher, our ROI analysis shows break-even in 18 months. Additionally, our solutions work 24/7 without breaks or turnover issues."
                        },
                        {
                            "objection": "Concerns about technical complexity",
                            "response": "Our implementation team handles all integration, with typical deployment taking under 3 months. We also provide comprehensive training and 24/7 support."
                        }
                    ],
                    "next_steps": [
                        "Schedule on-site assessment",
                        "Arrange demo with operations team",
                        "Share detailed implementation timeline",
                        "Introduce to reference customer"
                    ],
                    "recent_developments": "ABC just announced Q2 earnings with 15% growth. CEO specifically mentioned automation as a priority for the coming fiscal year. Recently hired a new Director of Supply Chain.",
                    "competitive_insights": "Currently evaluating competitor ForkliftAI's solution but has concerns about their limited integration capabilities with SAP.",
                    "value_propositions": [
                        "40% reduction in labor costs",
                        "25% increase in warehouse throughput",
                        "99.8% operational reliability",
                        "Seamless integration with existing systems",
                        "Scalable solution that grows with business needs"
                    ]
                }
            ]
        }
        return json.dumps(example, indent=2)
