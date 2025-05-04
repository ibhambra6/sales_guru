from typing import List, Optional
from pydantic import BaseModel, Field


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


class CompanyProfile(BaseModel):
    """Company information for prospect research"""
    industry: str
    size: str
    revenue: Optional[str] = Field(default="Information not readily available")
    funding: Optional[str] = Field(default="Information not readily available")
    growth_trajectory: str
    tech_stack: Optional[str] = Field(default="Information not readily available")
    products_services: str


class DecisionMakerProfile(BaseModel):
    """Decision maker information for prospect research"""
    role: str
    responsibilities: str
    career_history: Optional[str] = Field(default="Information not readily available")
    educational_background: Optional[str] = Field(default="Information not readily available")
    social_media_presence: Optional[str] = Field(default="Information not readily available")


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


class CallBrief(BaseModel):
    """Structured output for Sales Call Preparation Agent"""
    lead_name: str
    company_name: str
    classification: str = Field(..., pattern="^(HOT|WARM)$")
    company_snapshot: str
    decision_maker_profile: str
    pain_points: List[str]
    talking_points: List[str]
    next_steps: List[str]
    recent_developments: str
    competitive_insights: Optional[str] = Field(default=None)
    value_propositions: List[str]


class LeadQualificationResponse(BaseModel):
    """Container for multiple lead qualification records"""
    qualified_leads: List[LeadQualification]


class ProspectResearchResponse(BaseModel):
    """Container for multiple prospect research records"""
    enriched_leads: List[ProspectResearch]


class EmailOutreachResponse(BaseModel):
    """Container for multiple email templates"""
    email_templates: List[EmailTemplate]


class SalesCallPrepResponse(BaseModel):
    """Container for multiple call briefs"""
    call_briefs: List[CallBrief] 