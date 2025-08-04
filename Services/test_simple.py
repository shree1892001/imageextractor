#!/usr/bin/env python3
"""
Simple test to verify the visualization works with proper data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.JDVsCV_Enhanced import EnhancedCVJDMatcher
import plotly.graph_objects as go

def test_with_mock_data():
    """Test visualization with realistic mock data."""
    
    print("🧪 Testing Visualization with Mock Data")
    print("=" * 50)
    
    # Create a mock matcher instance
    matcher = EnhancedCVJDMatcher.__new__(EnhancedCVJDMatcher)
    
    # Realistic mock data that should work
    mock_results = {
        "basic_matching": {
            "match_score": 85,
            "matching_skills": ["JavaScript", "React", "Node.js", "Python", "Docker"],
            "missing_requirements": ["AWS", "Azure", "Leadership"],
            "match_category": "Good Match",
            "overall_assessment": "Strong technical match with some leadership gaps"
        },
        "culture_fit": {
            "culture_fit_score": 78,
            "communication_style_match": "Excellent",
            "work_style_alignment": "Good",
            "values_compatibility": "High",
            "team_collaboration_potential": "Strong",
            "cultural_risks": ["None identified"],
            "overall_culture_assessment": "Good cultural fit"
        },
        "career_trajectory": {
            "career_growth_potential": 82,
            "predicted_next_roles": ["Senior Developer", "Tech Lead"],
            "skill_evolution_timeline": "2-3 years",
            "industry_relevance_score": 88,
            "future_skill_requirements": ["Cloud Architecture", "Leadership"],
            "career_risk_factors": ["Market volatility"],
            "growth_recommendations": ["Certifications", "Leadership training"]
        },
        "market_intelligence": {
            "market_score": 75,
            "salary_range": {"min": 60000, "max": 120000, "median": 85000},
            "skill_demand": {"python": 85, "javascript": 78, "react": 72},
            "market_trends": {"growing_demand": True, "skill_evolution": "Rapid"}
        }
    }
    
    print("📊 Testing with realistic mock data:")
    for module, data in mock_results.items():
        print(f"  {module}: {type(data)}")
        if isinstance(data, dict):
            print(f"    Keys: {list(data.keys())}")
    
    # Test each visualization function
    print("\n📈 Testing individual charts:")
    
    # Test match score chart
    print("\n1. Match Score Chart:")
    try:
        fig1 = matcher._create_match_score_chart(mock_results)
        print(f"  ✅ Created successfully")
        print(f"  📊 Traces: {len(fig1.data)}")
        if len(fig1.data) > 0:
            print(f"  📊 Data points: {len(fig1.data[0].x)}")
            print(f"  📊 Values: {fig1.data[0].y}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test radar chart
    print("\n2. Radar Chart:")
    try:
        fig2 = matcher._create_radar_chart(mock_results)
        print(f"  ✅ Created successfully")
        print(f"  📊 Traces: {len(fig2.data)}")
        if len(fig2.data) > 0:
            print(f"  📊 Categories: {fig2.data[0].theta}")
            print(f"  📊 Values: {fig2.data[0].r}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test skills comparison chart
    print("\n3. Skills Comparison Chart:")
    try:
        fig3 = matcher._create_skills_comparison_chart(mock_results)
        print(f"  ✅ Created successfully")
        print(f"  📊 Traces: {len(fig3.data)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test module performance chart
    print("\n4. Module Performance Chart:")
    try:
        fig4 = matcher._create_module_performance_chart(mock_results)
        print(f"  ✅ Created successfully")
        print(f"  📊 Traces: {len(fig4.data)}")
        if len(fig4.data) > 0:
            print(f"  📊 Data points: {len(fig4.data[0].x)}")
            print(f"  📊 Values: {fig4.data[0].y}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test full dashboard
    print("\n5. Full Dashboard:")
    try:
        dashboard = matcher.create_visual_dashboard(mock_results)
        print(f"  ✅ Dashboard created successfully")
        print(f"  📊 Charts: {list(dashboard.keys())}")
        for chart_name, fig in dashboard.items():
            print(f"    📊 {chart_name}: {len(fig.data)} traces")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print("\n✅ Mock data test completed!")

if __name__ == "__main__":
    test_with_mock_data() 