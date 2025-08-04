#!/usr/bin/env python3
"""
Test script for visualization functions in JDVsCV_Enhanced.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.JDVsCV_Enhanced import EnhancedCVJDMatcher
import plotly.graph_objects as go

def test_visualization_functions():
    """Test the visualization functions with different data types."""
    
    # Create a mock matcher instance (without API key for testing)
    matcher = EnhancedCVJDMatcher.__new__(EnhancedCVJDMatcher)
    
    # Test data - string responses (simulating LLM responses)
    test_results_string = {
        "basic_matching": '{"match_score": "85%", "matching_skills": ["Python", "React", "JavaScript"], "missing_requirements": ["AWS", "Docker"], "match_category": "Good Match", "overall_assessment": "Strong technical match"}',
        "culture_fit": '{"culture_fit_score": "78%", "communication_style_match": "Excellent", "work_style_alignment": "Good", "values_compatibility": "High", "team_collaboration_potential": "Strong", "cultural_risks": ["None identified"], "overall_culture_assessment": "Good cultural fit"}',
        "career_trajectory": '{"career_growth_potential": "82%", "predicted_next_roles": ["Senior Developer", "Tech Lead"], "skill_evolution_timeline": "2-3 years", "industry_relevance_score": "88%", "future_skill_requirements": ["Cloud Architecture", "Leadership"], "career_risk_factors": ["Market volatility"], "growth_recommendations": ["Certifications", "Leadership training"]}'
    }
    
    # Test data - dictionary responses
    test_results_dict = {
        "basic_matching": {"match_score": 85, "matching_skills": ["Python", "React", "JavaScript"], "missing_requirements": ["AWS", "Docker"], "match_category": "Good Match", "overall_assessment": "Strong technical match"},
        "culture_fit": {"culture_fit_score": 78, "communication_style_match": "Excellent", "work_style_alignment": "Good", "values_compatibility": "High", "team_collaboration_potential": "Strong", "cultural_risks": ["None identified"], "overall_culture_assessment": "Good cultural fit"},
        "career_trajectory": {"career_growth_potential": 82, "predicted_next_roles": ["Senior Developer", "Tech Lead"], "skill_evolution_timeline": "2-3 years", "industry_relevance_score": 88, "future_skill_requirements": ["Cloud Architecture", "Leadership"], "career_risk_factors": ["Market volatility"], "growth_recommendations": ["Certifications", "Leadership training"]}
    }
    
    # Test data - mixed responses
    test_results_mixed = {
        "basic_matching": {"match_score": 85, "matching_skills": ["Python", "React", "JavaScript"], "missing_requirements": ["AWS", "Docker"], "match_category": "Good Match", "overall_assessment": "Strong technical match"},
        "culture_fit": '{"culture_fit_score": "78%", "communication_style_match": "Excellent", "work_style_alignment": "Good", "values_compatibility": "High", "team_collaboration_potential": "Strong", "cultural_risks": ["None identified"], "overall_culture_assessment": "Good cultural fit"}',
        "career_trajectory": {"career_growth_potential": 82, "predicted_next_roles": ["Senior Developer", "Tech Lead"], "skill_evolution_timeline": "2-3 years", "industry_relevance_score": 88, "future_skill_requirements": ["Cloud Architecture", "Leadership"], "career_risk_factors": ["Market volatility"], "growth_recommendations": ["Certifications", "Leadership training"]}
    }
    
    # Test data - empty/invalid responses
    test_results_empty = {
        "basic_matching": "Invalid response",
        "culture_fit": "Error in analysis",
        "career_trajectory": {}
    }
    
    print("Testing visualization functions...")
    
    # Test with string responses
    print("\n1. Testing with string responses:")
    try:
        fig1 = matcher._create_match_score_chart(test_results_string)
        print("✓ Match score chart created successfully")
        print(f"  - Chart has {len(fig1.data)} traces")
        print(f"  - Chart layout title: {fig1.layout.title.text}")
        if len(fig1.data) > 0:
            print(f"  - Data points: {len(fig1.data[0].x)}")
            print(f"  - Values: {fig1.data[0].y}")
    except Exception as e:
        print(f"✗ Match score chart failed: {e}")
    
    # Test with dictionary responses
    print("\n2. Testing with dictionary responses:")
    try:
        fig2 = matcher._create_match_score_chart(test_results_dict)
        print("✓ Match score chart created successfully")
        print(f"  - Chart has {len(fig2.data)} traces")
        print(f"  - Chart layout title: {fig2.layout.title.text}")
        if len(fig2.data) > 0:
            print(f"  - Data points: {len(fig2.data[0].x)}")
            print(f"  - Values: {fig2.data[0].y}")
    except Exception as e:
        print(f"✗ Match score chart failed: {e}")
    
    # Test with mixed responses
    print("\n3. Testing with mixed responses:")
    try:
        fig3 = matcher._create_match_score_chart(test_results_mixed)
        print("✓ Match score chart created successfully")
        print(f"  - Chart has {len(fig3.data)} traces")
        print(f"  - Chart layout title: {fig3.layout.title.text}")
        if len(fig3.data) > 0:
            print(f"  - Data points: {len(fig3.data[0].x)}")
            print(f"  - Values: {fig3.data[0].y}")
    except Exception as e:
        print(f"✗ Match score chart failed: {e}")
    
    # Test with empty/invalid responses
    print("\n4. Testing with empty/invalid responses:")
    try:
        fig4 = matcher._create_match_score_chart(test_results_empty)
        print("✓ Match score chart created successfully (should show no data message)")
        print(f"  - Chart has {len(fig4.data)} traces")
        print(f"  - Chart layout title: {fig4.layout.title.text}")
    except Exception as e:
        print(f"✗ Match score chart failed: {e}")
    
    # Test skills comparison chart
    print("\n5. Testing skills comparison chart:")
    try:
        fig5 = matcher._create_skills_comparison_chart(test_results_dict)
        print("✓ Skills comparison chart created successfully")
        print(f"  - Chart has {len(fig5.data)} traces")
        print(f"  - Chart layout title: {fig5.layout.title.text}")
    except Exception as e:
        print(f"✗ Skills comparison chart failed: {e}")
    
    # Test module performance chart
    print("\n6. Testing module performance chart:")
    try:
        fig6 = matcher._create_module_performance_chart(test_results_dict)
        print("✓ Module performance chart created successfully")
        print(f"  - Chart has {len(fig6.data)} traces")
        print(f"  - Chart layout title: {fig6.layout.title.text}")
        if len(fig6.data) > 0:
            print(f"  - Data points: {len(fig6.data[0].x)}")
            print(f"  - Values: {fig6.data[0].y}")
    except Exception as e:
        print(f"✗ Module performance chart failed: {e}")
    
    # Test radar chart
    print("\n7. Testing radar chart:")
    try:
        fig7 = matcher._create_radar_chart(test_results_dict)
        print("✓ Radar chart created successfully")
        print(f"  - Chart has {len(fig7.data)} traces")
        print(f"  - Chart layout title: {fig7.layout.title.text}")
        if len(fig7.data) > 0:
            print(f"  - Categories: {fig7.data[0].theta}")
            print(f"  - Values: {fig7.data[0].r}")
    except Exception as e:
        print(f"✗ Radar chart failed: {e}")
    
    print("\n✅ All visualization tests completed!")

if __name__ == "__main__":
    test_visualization_functions() 