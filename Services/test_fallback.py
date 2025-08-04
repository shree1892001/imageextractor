#!/usr/bin/env python3
"""
Test script to verify fallback system works with empty/invalid LLM responses
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.JDVsCV_Enhanced import EnhancedCVJDMatcher
from Common.constants import API_KEY

async def test_fallback_system():
    """Test the fallback system with empty/invalid LLM responses."""
    
    print("🧪 Testing Fallback System")
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
    
    # Test fallback data generation
    print("\n📊 Testing fallback data generation:")
    matcher = EnhancedCVJDMatcher.__new__(EnhancedCVJDMatcher)
    fallback_data = matcher._get_fallback_mock_data(mock_cv_text, mock_jd_text)
    
    for module_name, data in fallback_data.items():
        print(f"  {module_name}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {data}")
    
    # Test with real analysis (simulating LLM failures)
    print(f"\n🔍 Testing with real analysis (expecting fallback):")
    
    # Create matcher instance
    matcher = EnhancedCVJDMatcher(
        api_key=API_KEY,
        enabled_modules=["basic_matching", "culture_fit", "career_trajectory"]
    )
    
    # Test with real files if they exist
    cv_path = "E:\\imageextractor\\imageextractor\\React_JS_5_Years.pdf"
    jd_path = "E:\\imageextractor\\imageextractor\\React_Resume.pdf"
    
    if os.path.exists(cv_path) and os.path.exists(jd_path):
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
                    if "error" in result:
                        print(f"    - Error: {result['error']}")
                else:
                    print(f"    - Value: {str(result)[:100]}...")
            
            # Test visualization with results
            print(f"\n📈 Testing visualization with results:")
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
    
    print("\n✅ Fallback system test completed!")

if __name__ == "__main__":
    asyncio.run(test_fallback_system()) 