#!/usr/bin/env python3
"""
Debug script to test analysis modules and visualization with mock data
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.JDVsCV_Enhanced import EnhancedCVJDMatcher
from Common.constants import API_KEY

async def debug_analysis():
    """Debug the analysis modules with mock data."""
    
    print("🔍 Debugging Analysis Modules")
    print("=" * 50)
    
    # Create mock CV and JD text
    mock_cv_text = """
    EXPERIENCE
    Senior Software Engineer
    Tech Company Inc. | 2020 - Present
    • Developed full-stack web applications using React, Node.js, and Python
    • Led team of 5 developers in agile environment
    • Implemented CI/CD pipelines and automated testing
    • Technologies: JavaScript, Python, React, Node.js, AWS, Docker
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2016 - 2020
    """
    
    mock_jd_text = """
    SENIOR SOFTWARE ENGINEER
    We are looking for a Senior Software Engineer with:
    • 5+ years experience in full-stack development
    • Strong knowledge of JavaScript, React, Node.js
    • Experience with cloud platforms (AWS, Azure)
    • Familiarity with Docker and CI/CD
    • Leadership experience preferred
    • Python knowledge is a plus
    """
    
    # Test with mock data first
    print("\n📊 Testing with mock data:")
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
        }
    }
    
    # Create matcher instance
    matcher = EnhancedCVJDMatcher(
        api_key=API_KEY,
        enabled_modules=["basic_matching", "culture_fit", "career_trajectory"]
    )
    
    # Test visualization with mock data
    print("\n📈 Testing visualization with mock data:")
    try:
        dashboard = matcher.create_visual_dashboard(mock_results)
        print("✅ Dashboard created successfully")
        
        for chart_name, fig in dashboard.items():
            print(f"  📊 {chart_name}: {len(fig.data)} traces")
            if len(fig.data) > 0:
                if hasattr(fig.data[0], 'x'):
                    print(f"    - Data points: {len(fig.data[0].x)}")
                if hasattr(fig.data[0], 'y'):
                    print(f"    - Values: {fig.data[0].y}")
                if hasattr(fig.data[0], 'r'):
                    print(f"    - Radar values: {fig.data[0].r}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
    
    # Test with real analysis (if files exist)
    cv_path = "E:\\imageextractor\\imageextractor\\React_JS_5_Years.pdf"
    jd_path = "E:\\imageextractor\\imageextractor\\React_Resume.pdf"
    
    if os.path.exists(cv_path) and os.path.exists(jd_path):
        print(f"\n🔍 Testing with real analysis:")
        print(f"  CV Path: {cv_path}")
        print(f"  JD Path: {jd_path}")
        
        try:
            # Run analysis
            results = await matcher.run_comprehensive_analysis(
                cv_path=cv_path,
                jd_path=jd_path,
                company_values="Innovation, Collaboration, Continuous Learning"
            )
            
            print(f"\n📊 Analysis Results:")
            for module_name, result in results.items():
                print(f"  {module_name}: {type(result)}")
                if isinstance(result, dict):
                    print(f"    - Keys: {list(result.keys())}")
                    if "match_score" in result:
                        print(f"    - Match Score: {result['match_score']}")
                    if "culture_fit_score" in result:
                        print(f"    - Culture Score: {result['culture_fit_score']}")
                    if "career_growth_potential" in result:
                        print(f"    - Career Score: {result['career_growth_potential']}")
                else:
                    print(f"    - Value: {str(result)[:100]}...")
            
            # Test visualization with real results
            print(f"\n📈 Testing visualization with real results:")
            dashboard = matcher.create_visual_dashboard(results)
            
            for chart_name, fig in dashboard.items():
                print(f"  📊 {chart_name}: {len(fig.data)} traces")
                if len(fig.data) > 0:
                    if hasattr(fig.data[0], 'x'):
                        print(f"    - Data points: {len(fig.data[0].x)}")
                    if hasattr(fig.data[0], 'y'):
                        print(f"    - Values: {fig.data[0].y}")
                    if hasattr(fig.data[0], 'r'):
                        print(f"    - Radar values: {fig.data[0].r}")
                        
        except Exception as e:
            print(f"❌ Real analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  PDF files not found, skipping real analysis test")
        print(f"  CV Path exists: {os.path.exists(cv_path)}")
        print(f"  JD Path exists: {os.path.exists(jd_path)}")
    
    print("\n✅ Debug analysis completed!")

if __name__ == "__main__":
    asyncio.run(debug_analysis()) 