from typing import List, Dict
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import json

class LLMSelector:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def get_selectors(self, prompt: str, context: Dict) -> List[str]:
        """Generate selectors using Gemini model"""
        try:
            # Format the context for better prompt
            context_str = self._format_context(context)
            
            full_prompt = f"""
            Page Context:
            {context_str}

            Task:
            {prompt}

            Return ONLY a JSON array of the 5 most likely CSS selectors to accomplish this task, ordered by specificity and reliability.
            Focus on generating precise selectors that uniquely identify the target element.
            Consider visible elements first, then fall back to general selectors.
            Example format: ["selector1", "selector2", "selector3", "selector4", "selector5"]
            """

            response = self.model.generate_content(full_prompt)
            return self._extract_selectors(response.text)
        except Exception as e:
            print(f"Error generating selectors: {e}")
            return []

    def _format_context(self, context: Dict) -> str:
        """Format context for LLM prompt"""
        formatted_parts = []
        
        for key, value in context.items():
            if isinstance(value, str):
                # Try to parse JSON strings
                try:
                    parsed_value = json.loads(value)
                    formatted_parts.append(f"{key}:\n{json.dumps(parsed_value, indent=2)}")
                except:
                    formatted_parts.append(f"{key}: {value}")
            else:
                formatted_parts.append(f"{key}: {value}")
        
        return "\n\n".join(formatted_parts)

    def _extract_selectors(self, response_text: str) -> List[str]:
        """
        Extract selector list from model response
        """
        try:
            # Remove markdown formatting if present
            clean_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Basic parsing to extract array
            start_idx = clean_text.find('[')
            end_idx = clean_text.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                return []
                
            selector_text = clean_text[start_idx:end_idx + 1]
            
            # Convert string representation to actual list
            import ast
            selectors = ast.literal_eval(selector_text)
            
            # Validate and clean selectors
            return [s.strip() for s in selectors if isinstance(s, str) and s.strip()]
            
        except Exception as e:
            print(f"Error extracting selectors: {str(e)}")
            return []

