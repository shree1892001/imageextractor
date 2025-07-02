"""
Utility functions for the Voice Assistant application.
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional


def setup_logging(level=logging.INFO):
    """Set up logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('voice_assistant')


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text"""
    try:
        json_str = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_str:
            return None
        
        json_str = json_str.group(0)
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
        return None


def filter_html(html: str, max_length: int = 3000) -> str:
    """Filter HTML to make it more readable for LLM processing"""
    filtered = re.sub(
        r'<(input|button|a|form|select|textarea|div|ul|li)[^>]*>',
        lambda m: m.group(0) + '\n',
        html
    )
    return filtered[:max_length]


def find_best_match(target: str, options: List[str]) -> Optional[str]:
    """Find the best matching option from available options"""
    if not options:
        return None
        
    target = target.lower()
    
    # Direct match
    for option in options:
        if option.lower() == target:
            return option
    
    # Partial match
    for option in options:
        if target in option.lower():
            return option
    
    # Handle special cases for counties
    if 'county' in target:
        county_name = target.replace('county', '').strip()
        for option in options:
            if county_name in option.lower():
                return option
    
    return None


def format_address(address_parts: Dict[str, str]) -> str:
    """Format address parts into a single string"""
    parts = []
    
    if address_parts.get('street'):
        parts.append(address_parts['street'])
    
    city_state_zip = []
    if address_parts.get('city'):
        city_state_zip.append(address_parts['city'])
    
    if address_parts.get('state'):
        if city_state_zip:
            city_state_zip.append(f", {address_parts['state']}")
        else:
            city_state_zip.append(address_parts['state'])
    
    if address_parts.get('zip'):
        city_state_zip.append(f" {address_parts['zip']}")
    
    if city_state_zip:
        parts.append(''.join(city_state_zip))
    
    return ', '.join(parts)
