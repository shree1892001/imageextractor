#!/usr/bin/env python3
"""
Test script to debug career trajectory JSON parsing
"""

import re
import json

def test_career_parsing():
    """Test career trajectory JSON parsing with various corrupted responses."""
    
    print("🧪 Testing Career Trajectory JSON Parsing")
    print("=" * 50)
    
    # Test cases with corrupted career trajectory JSON responses
    test_cases = [
        {
            "name": "Corrupted career JSON with newlines",
            "json": '{\n  "career_growth_potential": 85,\n  "predicted_next_roles": ["Senior Developer", "Tech Lead"],\n  "skill_evolution_timeline": "2-3 years",\n  "industry_relevance_score": 90,\n  "future_skill_requirements": ["Cloud Architecture", "Leadership"],\n  "career_risk_factors": ["Market volatility"],\n  "growth_recommendations": ["Continuous learning", "Leadership development"]\n}',
            "expected_fields": ["career_growth_potential", "predicted_next_roles", "skill_evolution_timeline"]
        },
        {
            "name": "Partial career JSON",
            "json": '{"career_growth_potential": 80, "predicted_next_roles": ["Senior Developer", "Tech Lead"], "skill_evolution_timeline": "2-3 years"',
            "expected_fields": ["career_growth_potential", "predicted_next_roles", "skill_evolution_timeline"]
        },
        {
            "name": "Empty response",
            "json": "",
            "expected_fields": []
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
        
        return json_str
    
    def _extract_partial_career_json(json_str: str) -> dict:
        """Extract partial data from corrupted career trajectory JSON."""
        try:
            # Try to find and extract individual fields
            result = {}
            
            # Extract career_growth_potential
            growth_match = re.search(r'"career_growth_potential"\s*:\s*(\d+)', json_str)
            if growth_match:
                result["career_growth_potential"] = int(growth_match.group(1))
            
            # Extract predicted_next_roles
            roles_match = re.search(r'"predicted_next_roles"\s*:\s*\[(.*?)\]', json_str)
            if roles_match:
                roles_str = roles_match.group(1)
                # Extract individual roles
                roles = re.findall(r'"([^"]+)"', roles_str)
                result["predicted_next_roles"] = roles
            
            # Extract skill_evolution_timeline
            timeline_match = re.search(r'"skill_evolution_timeline"\s*:\s*"([^"]+)"', json_str)
            if timeline_match:
                result["skill_evolution_timeline"] = timeline_match.group(1)
            
            # Extract industry_relevance_score
            relevance_match = re.search(r'"industry_relevance_score"\s*:\s*(\d+)', json_str)
            if relevance_match:
                result["industry_relevance_score"] = int(relevance_match.group(1))
            
            # Extract future_skill_requirements
            skills_match = re.search(r'"future_skill_requirements"\s*:\s*\[(.*?)\]', json_str)
            if skills_match:
                skills_str = skills_match.group(1)
                # Extract individual skills
                skills = re.findall(r'"([^"]+)"', skills_str)
                result["future_skill_requirements"] = skills
            
            # Extract career_risk_factors
            risks_match = re.search(r'"career_risk_factors"\s*:\s*\[(.*?)\]', json_str)
            if risks_match:
                risks_str = risks_match.group(1)
                # Extract individual risks
                risks = re.findall(r'"([^"]+)"', risks_str)
                result["career_risk_factors"] = risks
            
            # Extract growth_recommendations
            recs_match = re.search(r'"growth_recommendations"\s*:\s*\[(.*?)\]', json_str)
            if recs_match:
                recs_str = recs_match.group(1)
                # Extract individual recommendations
                recs = re.findall(r'"([^"]+)"', recs_str)
                result["growth_recommendations"] = recs
            
            # If we extracted any data, return it
            if result:
                # Ensure we have at least a career_growth_potential
                if "career_growth_potential" not in result:
                    result["career_growth_potential"] = 80  # Default score
                if "predicted_next_roles" not in result:
                    result["predicted_next_roles"] = ["Senior Developer", "Tech Lead"]
                if "skill_evolution_timeline" not in result:
                    result["skill_evolution_timeline"] = "2-3 years"
                if "industry_relevance_score" not in result:
                    result["industry_relevance_score"] = 85
                if "future_skill_requirements" not in result:
                    result["future_skill_requirements"] = ["Cloud Architecture", "Leadership"]
                if "career_risk_factors" not in result:
                    result["career_risk_factors"] = ["Market volatility"]
                if "growth_recommendations" not in result:
                    result["growth_recommendations"] = ["Continuous learning", "Leadership development"]
                
                return result
            
        except Exception as e:
            print(f"❌ Failed to extract partial career JSON: {e}")
        
        return {}
    
    print("📊 Testing career trajectory JSON parsing:")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Input: {test_case['json'][:100]}...")
        
        try:
            # Test the fix_json_string method
            fixed_json = _fix_json_string(test_case['json'])
            print(f"   Fixed JSON: {fixed_json[:100]}...")
            
            # Test the extract_partial_career_json method
            partial_data = _extract_partial_career_json(test_case['json'])
            print(f"   Extracted data: {partial_data}")
            
            # Check if expected fields were extracted
            if test_case['expected_fields']:
                extracted_fields = list(partial_data.keys())
                missing_fields = [field for field in test_case['expected_fields'] if field not in extracted_fields]
                if missing_fields:
                    print(f"   ⚠️  Missing fields: {missing_fields}")
                else:
                    print(f"   ✅ All expected fields extracted")
            else:
                if not partial_data:
                    print(f"   ✅ Correctly handled empty response")
                else:
                    print(f"   ⚠️  Unexpected data extracted from empty response")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Career trajectory JSON parsing tests completed!")

if __name__ == "__main__":
    test_career_parsing() 