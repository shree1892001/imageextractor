"""
LLM integration service.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional

import google.generativeai as genai

from Core.base import LLMService
from Core.config import Config
from Core.utils import extract_json_from_text


class GeminiService(LLMService):
    """Service for Gemini LLM integration"""
    
    def __init__(self, config: Config):
        """Initialize the LLM service"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.generation_config = {
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.95),
            "top_k": config.get("top_k", 40),
            "max_output_tokens": config.get("max_output_tokens", 4096),
        }
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    
    def initialize(self) -> bool:
        """Initialize the LLM service"""
        try:
            api_key = self.config.get("gemini_api_key")
            if not api_key:
                self.logger.error("No API key provided for Gemini")
                return False
            
            genai.configure(api_key=api_key)
            model_name = self.config.get("llm_model", "gemini-1.5-flash")
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            self.logger.info(f"LLM service initialized with model {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing LLM service: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the LLM service"""
        try:
            # No specific cleanup needed for Gemini
            self.logger.info("LLM service shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Error shutting down LLM service: {e}")
            return False
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the LLM"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"
    
    def get_structured_guidance(self, prompt: str) -> Dict[str, Any]:
        """Get structured guidance from the LLM"""
        try:
            response = self.model.generate_content(prompt)
            result = extract_json_from_text(response.text)
            if result:
                return result
            
            # If JSON extraction failed, try to parse as a list
            list_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if list_match:
                list_str = list_match.group(0)
                try:
                    return {"selectors": json.loads(list_str)}
                except:
                    pass
            
            # Return a default structure if all parsing fails
            return {"error": "Failed to parse structured response", "raw_response": response.text}
        except Exception as e:
            self.logger.error(f"Error getting structured guidance: {e}")
            return {"error": str(e)}
    
    def get_selectors(self, task: str, context: Dict[str, Any]) -> List[str]:
        """Get selectors for a specific task based on page context"""
        prompt = f"""
Based on the current web page context, generate the 5 most likely CSS selectors to {task}.
Focus on precise selectors that would uniquely identify the element.

Current Page:
Title: {context.get('title', 'N/A')}
URL: {context.get('url', 'N/A')}

Input Fields Found:
{self._format_input_fields(context.get('input_fields', []))}

Menu Items Found:
{self._format_menu_items(context.get('menu_items', []))}

Relevant HTML:
{context.get('html', '')[:1000]}

IMPORTANT: If this appears to be a PrimeNG component (classes containing p-dropdown, p-component, etc.),
prioritize selectors that target PrimeNG specific elements:
- Dropdown: .p-dropdown, .p-dropdown-trigger
- Panel: .p-dropdown-panel
- Items: .p-dropdown-item, .p-dropdown-items li
- Filter: .p-dropdown-filter

Respond ONLY with a JSON array of selector strings. Example:
["selector1", "selector2", "selector3", "selector4", "selector5"]
"""
        
        try:
            response = self.generate_content(prompt)
            selectors_match = re.search(r'\[.*\]', response, re.DOTALL)
            if selectors_match:
                selectors_json = selectors_match.group(0)
                selectors = json.loads(selectors_json)
                return selectors[:5]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Selector generation error: {e}")
            return []
    
    def _format_input_fields(self, input_fields: List[Dict[str, str]]) -> str:
        """Format input fields for LLM prompt"""
        result = ""
        for idx, field in enumerate(input_fields):
            result += f"{idx + 1}. {field.get('tag', 'input')} - "
            result += f"type: {field.get('type', '')}, "
            result += f"id: {field.get('id', '')}, "
            result += f"name: {field.get('name', '')}, "
            result += f"placeholder: {field.get('placeholder', '')}, "
            result += f"aria-label: {field.get('aria-label', '')}\n"
        return result
    
    def _format_menu_items(self, menu_items: List[Dict[str, Any]]) -> str:
        """Format menu items for LLM prompt"""
        result = ""
        for idx, item in enumerate(menu_items):
            submenu_indicator = " (has submenu)" if item.get("has_submenu") else ""
            result += f"{idx + 1}. {item.get('text', '')}{submenu_indicator}\n"
        return result
