#!/usr/bin/env python3
"""
Test script to verify JSON parsing improvements for the CV-JD matcher.
"""

import json
import re
from imageextractor.Services.JDVsCV_Enhanced import AnalysisModule

class TestAnalysisModule(AnalysisModule):
    """Test module to verify JSON parsing."""
    
    def get_name(self) -> str:
        return "test"
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        return {}

def test_json_parsing():
    """Test the JSON parsing improvements."""
    
    # Create a test instance
    test_module = TestAnalysisModule()
    
    # Test case 1: The specific error you encountered
    problematic_json = '''{
 "match_score": 95,
 "matching_skills": ["React.js", "Redux", "React Hooks", "HTML5", "CSS3", "JavaScript (ES6+)", "TypeScript", "RESTful APIs", "GraphQL", "Git", "Jest", "Enzyme", "Webpack", "Babel", "Agile Methodologies", "Responsive Design", "Mobile-First Development", "Material-UI", "Ant Design", "CI/CD"],
 "missing_requirements": ["Experience with Next.js or other SSR frameworks", "Familiarity with cloud platforms like AWS, Azure, or Google Cloud", "Understanding of CI/CD pipelines (although some experience is mentioned, a deeper understanding might be needed)"],
 "match_category": "Best Match",
 "overall_assessment": "John Doe\\'s resume is an excellent match for the React Developer position. He possesses all the core required skills and a significant amount of the preferred skills. His 5+ years of experience, detailed project descriptions, and demonstrable achievements in performance optimization and team leadership strongly support his candidacy. While he doesn\\'t explicitly mention experience with Next.js or cloud platforms, his extensive React experience and proven ability to quickly learn and adapt suggest he could readily acquire these skills. The minor gaps in preferred skills are not significant enough to detract substantially from his strong overall profile."
}'''
    
    print("🔍 Testing JSON parsing improvements...")
    print(f"📝 Original JSON length: {len(problematic_json)}")
    print(f"📝 Original JSON preview: {problematic_json[:200]}...")
    
    # Test the fix_json_string method
    try:
        fixed_json = test_module._fix_json_string(problematic_json)
        print(f"✅ JSON fixed successfully")
        print(f"📝 Fixed JSON length: {len(fixed_json)}")
        print(f"📝 Fixed JSON preview: {fixed_json[:200]}...")
        
        # Try to parse the fixed JSON
        parsed_result = json.loads(fixed_json)
        print(f"✅ JSON parsed successfully!")
        print(f"📊 Parsed data keys: {list(parsed_result.keys())}")
        print(f"📊 Match score: {parsed_result.get('match_score')}")
        print(f"📊 Skills count: {len(parsed_result.get('matching_skills', []))}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"📝 Attempting partial extraction...")
        
        # Test partial extraction
        partial_data = test_module._extract_partial_json(problematic_json)
        if partial_data:
            print(f"✅ Partial extraction successful!")
            print(f"📊 Extracted data: {partial_data}")
        else:
            print(f"❌ Partial extraction also failed")
    
    # Test case 2: Another problematic case
    test_json_2 = '''{"match_score": 85, "matching_skills": ["Python", "Django", "React"], "overall_assessment": "Candidate\\'s skills match well with the job requirements."}'''
    
    print(f"\n🔍 Testing second case...")
    try:
        fixed_json_2 = test_module._fix_json_string(test_json_2)
        parsed_result_2 = json.loads(fixed_json_2)
        print(f"✅ Second case parsed successfully!")
    except json.JSONDecodeError as e:
        print(f"❌ Second case failed: {e}")
        partial_data_2 = test_module._extract_partial_json(test_json_2)
        if partial_data_2:
            print(f"✅ Second case partial extraction successful!")
    
    print(f"\n✅ JSON parsing test completed!")

if __name__ == "__main__":
    test_json_parsing() 