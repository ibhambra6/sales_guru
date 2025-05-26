#!/usr/bin/env python3
"""
Test script to validate dynamic CSV functionality in Sales Guru
"""

import os
from src.sales_guru.supervisors import SupervisorAgentLogic

def test_csv_functionality():
    """Test that the supervisor can properly count leads from different CSV files"""
    
    # Test with the new test file
    test_csv_path = "test_leads.csv"
    
    if not os.path.exists(test_csv_path):
        print(f"❌ Test CSV file not found: {test_csv_path}")
        return False
    
    # Initialize supervisor with test CSV
    supervisor = SupervisorAgentLogic(csv_file_path=test_csv_path)
    
    # Test CSV lead counting
    lead_count = supervisor.count_csv_leads()
    print(f"✅ Successfully counted {lead_count} leads in {test_csv_path}")
    
    # Test CSV data loading
    csv_data = supervisor.get_csv_lead_data()
    print(f"✅ Successfully loaded {len(csv_data)} lead records")
    
    # Verify the data matches expected structure
    if len(csv_data) > 0:
        first_lead = csv_data[0]
        required_fields = ['Name', 'Company Name', 'Email', 'Phone']
        missing_fields = [field for field in required_fields if field not in first_lead]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            print("✅ CSV structure validation passed")
            print(f"   Sample lead: {first_lead['Name']} from {first_lead['Company Name']}")
    
    # Test with default CSV (if it exists)
    default_csv = "knowledge/leads.csv"
    if os.path.exists(default_csv):
        supervisor_default = SupervisorAgentLogic()  # Should use default path
        default_count = supervisor_default.count_csv_leads()
        print(f"✅ Default CSV has {default_count} leads")
    else:
        print(f"ℹ️  Default CSV not found: {default_csv}")
    
    print("\n🎉 All CSV functionality tests passed!")
    return True

if __name__ == "__main__":
    print("Testing Sales Guru Dynamic CSV Functionality")
    print("=" * 50)
    
    try:
        success = test_csv_functionality()
        if success:
            print("\n✅ All tests completed successfully!")
            print("\nYou can now use any CSV file with Sales Guru by:")
            print("1. Running: crewai run")
            print("2. Entering your CSV file path when prompted")
            print("3. Providing your company details")
        else:
            print("\n❌ Some tests failed. Please check the issues above.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 