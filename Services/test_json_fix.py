#!/usr/bin/env python3
"""
Test script to verify enhanced JSON fixing for the "Expecting property name" error
"""

import re
import json

def test_json_fixing():
    """Test the enhanced JSON fixing with the specific error case."""
    
    print("🧪 Testing Enhanced JSON Fixing")
    print("=" * 50)
    
    # Test the specific error case you're encountering
    test_cases = [
        {
            "name": "Your specific error case",
            "json": '{\n  "match_score": 95,\n  "matching_skills": ["React.js", "Redux", "React Hooks", "HTML5", "CSS3", ...',
            "expected_success": True
        },
        {
            "name": "Career trajectory with newlines",
            "json": '{\n  "career_growth_potential": 85,\n  "predicted_next_roles": ["Senior Developer", "Tech Lead"],\n  "skill_evolution_timeline": "2-3 years"\n}',
            "expected_success": True
        },
        {
            "name": "Culture fit with formatting",
            "json": '{\n  "culture_fit_score": 85,\n  "communication_style_match": "Excellent",\n  "work_style_alignment": "Good"\n}',
            "expected_success": True
        },
        {
            "name": "Malformed JSON with unquoted properties",
            "json": '{match_score: 95, matching_skills: ["React.js", "Redux"]}',
            "expected_success": True
        }
    ]
    
    def _fix_json_string(json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx != -1:
            json_str = json_str[start_idx:]
        
        # Remove any text after the last }
        end_idx = json_str.rfind('}')
        if end_idx != -1:
            json_str = json_str[:end_idx + 1]
        
        # Fix common issues
        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
        json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
        
        # Fix the specific "Expecting property name enclosed in double quotes" error
        # This usually happens when there are newlines or spaces after opening brace
        json_str = re.sub(r'^\s*{\s*', '{', json_str)  # Remove spaces after opening brace
        json_str = re.sub(r'\s*}\s*$', '}', json_str)  # Remove spaces before closing brace
        
        # Fix property names that might have spaces or newlines
        json_str = re.sub(r'(\w+)\s*:\s*', r'"\1":', json_str)  # Ensure property names are quoted
        
        # Fix common JSON formatting issues
        json_str = json_str.replace('\\n', ' ')  # Replace literal \n with space
        json_str = json_str.replace('\\r', ' ')  # Replace literal \r with space
        json_str = json_str.replace('\\t', ' ')  # Replace literal \t with space
        
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        return json_str
    
    print("📊 Testing JSON fixing with various corrupted responses:")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Input: {test_case['json'][:100]}...")
        
        try:
            # Test the original JSON (should fail)
            try:
                original_parsed = json.loads(test_case['json'])
                print(f"   ⚠️  Original JSON parsed successfully (unexpected)")
            except json.JSONDecodeError as e:
                print(f"   ❌ Original JSON failed: {e}")
            
            # Test the fixed JSON
            fixed_json = _fix_json_string(test_case['json'])
            print(f"   🔧 Fixed JSON: {fixed_json[:100]}...")
            
            try:
                parsed_result = json.loads(fixed_json)
                print(f"   ✅ Fixed JSON parsed successfully: {parsed_result}")
                if test_case['expected_success']:
                    print(f"   ✅ Expected success - PASS")
                else:
                    print(f"   ⚠️  Unexpected success")
            except json.JSONDecodeError as e:
                print(f"   ❌ Fixed JSON still failed: {e}")
                if not test_case['expected_success']:
                    print(f"   ✅ Expected failure - PASS")
                else:
                    print(f"   ❌ Unexpected failure")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ JSON fixing tests completed!")

if __name__ == "__main__":
    test_json_fixing() 