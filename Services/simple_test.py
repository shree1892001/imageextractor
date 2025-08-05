#!/usr/bin/env python3
"""
Simple test for JSON fixing.
"""

import json
import re

def fix_json_string(json_str: str) -> str:
    """Fix common JSON formatting issues with minimal intervention."""
    # Remove any text before the first { and after the last }
    start_idx = json_str.find('{')
    if start_idx != -1:
        json_str = json_str[start_idx:]
    
    end_idx = json_str.rfind('}')
    if end_idx != -1:
        json_str = json_str[:end_idx + 1]
    
    # Basic cleanup
    json_str = json_str.replace('\n', ' ').replace('\r', ' ')
    json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
    
    # Fix property names that aren't quoted
    json_str = re.sub(r'(\w+)\s*:\s*', r'"\1":', json_str)
    
    # Fix literal escape sequences
    json_str = json_str.replace('\\n', ' ')
    json_str = json_str.replace('\\r', ' ')
    json_str = json_str.replace('\\t', ' ')
    
    # Fix single quotes within string values (but be very careful)
    # Only fix single quotes that are clearly inside string values
    json_str = re.sub(r'([^\\])\'(?=[^"]*"[^"]*$)', r'\1\\\'', json_str)
    
    # Fix backslashes that aren't properly escaped
    json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', json_str)
    
    # Remove trailing commas before closing brackets/braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return json_str

# Test with a simple problematic JSON
test_json = '''{
    "test": "value with \\'single quotes\\'",
    "another": "test with \\"double quotes\\"",
    "array": ["item1", "item2", "item3"]
}'''

print("Testing JSON fixing...")
print("Original:", test_json)

try:
    parsed = json.loads(test_json)
    print("✅ Original JSON is valid!")
except json.JSONDecodeError as e:
    print(f"❌ Original JSON error: {e}")
    
    fixed = fix_json_string(test_json)
    print("Fixed:", fixed)
    
    try:
        parsed = json.loads(fixed)
        print("✅ Fixed JSON is valid!")
    except json.JSONDecodeError as e2:
        print(f"❌ Fixed JSON error: {e2}") 