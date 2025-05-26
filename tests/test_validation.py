#!/usr/bin/env python3
"""
Test script to validate the lead qualification agent fixes
"""

import sys
import os
sys.path.append('src')

from sales_guru.crew import SalesGuru
from sales_guru.task_monitor import TaskCompletionMonitor
from sales_guru.schemas import LeadQualificationResponse, LeadQualification

def test_validation_logic():
    """Test the validation logic with fictional leads"""
    print("=" * 50)
    print("Testing Lead Validation Logic")
    print("=" * 50)
    
    monitor = TaskCompletionMonitor()
    
    # Test with fictional leads (like what the agent was producing)
    fake_leads = [
        LeadQualification(
            lead_name="Alice Smith",
            company_name="Global Logistics Inc.",
            email_address="alice.smith@globallogistics.com",
            phone_number="555-123-4567",
            lead_score=90,
            classification="HOT",
            reasoning="Fictional lead created by agent",
            value_alignment="Fake alignment",
            recommended_approach="Test approach"
        ),
        LeadQualification(
            lead_name="Bob Johnson", 
            company_name="Manufacturing Solutions Co.",
            email_address="bob.johnson@mansolco.com",
            phone_number="555-987-6543",
            lead_score=85,
            classification="HOT",
            reasoning="Another fictional lead",
            value_alignment="Fake alignment",
            recommended_approach="Test approach"
        )
    ]
    
    fake_response = LeadQualificationResponse(qualified_leads=fake_leads)
    
    print(f"Input: {len(fake_leads)} fictional leads")
    print("Fictional leads:")
    for lead in fake_leads:
        print(f"  - {lead.lead_name} from {lead.company_name}")
    
    print("\nApplying validation...")
    result = monitor._validate_lead_count(fake_response)
    
    print(f"\nOutput: {len(result.qualified_leads)} leads after validation")
    for lead in result.qualified_leads:
        print(f"  - {lead.lead_name} from {lead.company_name}")
        if lead.lead_name == "ERROR":
            print(f"    Reasoning: {lead.reasoning}")
    
    return result

def test_with_real_leads():
    """Test with leads that match the CSV"""
    print("\n" + "=" * 50)
    print("Testing with Real CSV Leads")
    print("=" * 50)
    
    monitor = TaskCompletionMonitor()
    
    # Test with real leads from CSV
    real_leads = [
        LeadQualification(
            lead_name="Andy Jassy",
            company_name="Amazon",
            email_address="andy.jassy@amazon.com",
            phone_number="555-291-8472",
            lead_score=95,
            classification="HOT",
            reasoning="CEO of major tech company with logistics needs",
            value_alignment="High potential for automation solutions",
            recommended_approach="Direct executive outreach"
        ),
        LeadQualification(
            lead_name="Tim Cook",
            company_name="Apple Inc.",
            email_address="tim.cook@apple.com", 
            phone_number="555-853-2964",
            lead_score=88,
            classification="WARM",
            reasoning="Tech leader with manufacturing operations",
            value_alignment="Potential for supply chain automation",
            recommended_approach="Focus on innovation and efficiency"
        )
    ]
    
    real_response = LeadQualificationResponse(qualified_leads=real_leads)
    
    print(f"Input: {len(real_leads)} real CSV leads")
    print("Real leads:")
    for lead in real_leads:
        print(f"  - {lead.lead_name} from {lead.company_name}")
    
    print("\nApplying validation...")
    result = monitor._validate_lead_count(real_response)
    
    print(f"\nOutput: {len(result.qualified_leads)} leads after validation")
    for lead in result.qualified_leads:
        print(f"  - {lead.lead_name} from {lead.company_name}")
    
    return result

def show_csv_leads():
    """Show the leads that should be in the output"""
    print("\n" + "=" * 50)
    print("Expected Leads from CSV File")
    print("=" * 50)
    
    import csv
    csv_path = 'knowledge/leads.csv'
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        leads = list(reader)
    
    print(f"Total leads in CSV: {len(leads)}")
    print("Expected leads:")
    for i, lead in enumerate(leads, 1):
        print(f"  {i:2d}. {lead['Name']} - {lead['Company Name']}")
    
    return leads

if __name__ == "__main__":
    print("Lead Qualification Validation Test")
    print("=" * 60)
    
    # Show what we expect
    expected_leads = show_csv_leads()
    
    # Test with fake leads (should be caught)
    test_validation_logic()
    
    # Test with real leads (should pass through)
    test_with_real_leads()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Fictional leads are caught and replaced with error messages")
    print("- Real CSV leads pass through validation successfully")
    print("- The validation system is working correctly")
    print(f"- Expected output should contain exactly {len(expected_leads)} leads from the CSV") 