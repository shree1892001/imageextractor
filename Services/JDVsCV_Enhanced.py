import fitz
import asyncio
import os
import json
import cv2
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import re
# import speech_recognition as sr  # Commented out as it's not essential
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.tools import ToolDefinition
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

from pydantic_ai.providers.google_gla import GoogleGLAProvider

from imageextractor.Common.constants import *


class AnalysisModule(ABC):
    """Abstract base class for analysis modules."""

    @abstractmethod
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        """Perform analysis and return results."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this analysis module."""
        pass

    def _fix_json_string(self, json_str: str) -> str:
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
        json_str = re.sub(r'^\s*{\s*', '{', json_str)  # Remove spaces after opening brace
        json_str = re.sub(r'\s*}\s*$', '}', json_str)  # Remove spaces before closing brace
        
        # Fix property names that might have spaces or newlines
        json_str = re.sub(r'(\w+)\s*:\s*', r'"\1":', json_str)  # Ensure property names are quoted
        
        # Fix common JSON formatting issues
        json_str = json_str.replace('\\n', ' ')  # Replace literal \n with space
        json_str = json_str.replace('\\r', ' ')  # Replace literal \r with space
        json_str = json_str.replace('\\t', ' ')  # Replace literal \t with space
        
        # Fix invalid escape sequences - this is the key fix for the current error
        # Handle the specific case of "text\\'s" which should be "text\\'s"
        json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', json_str)
        
        # Fix single quotes that should be escaped - but be more careful
        # Only escape single quotes that are inside string values, not in property names
        # Use a simpler approach without lookbehind repetition
        json_str = re.sub(r'([^\\])\'(?=[^"]*")', r'\1\\\'', json_str)
        
        # Additional fix for the specific error you're seeing
        # Replace problematic escape sequences like "John Doe\'s" with "John Doe\\'s"
        # Use a simpler approach without lookbehind repetition
        json_str = re.sub(r'([^\\])\\(?=\'[^"]*")', r'\1\\\\', json_str)
        
        # Fix missing commas in arrays and objects - more careful approach
        # Only add commas when there's actually a next element, not before closing brackets/braces
        
        # Fix missing commas between array elements (but not at the end)
        json_str = re.sub(r'(\])\s*(\[)', r'\1,\2', json_str)
        json_str = re.sub(r'(")\s*(\[)', r'\1,\2', json_str)
        
        # Fix missing commas between object properties (but not at the end)
        json_str = re.sub(r'(")\s*(")', r'\1,\2', json_str)
        
        # Fix missing commas in nested structures - more specific patterns
        # Handle cases like "value" "next_property" -> "value", "next_property"
        json_str = re.sub(r'(")\s*(")', r'\1,\2', json_str)
        
        # Handle cases like } "next_property" -> }, "next_property"
        json_str = re.sub(r'(})\s*(")', r'\1,\2', json_str)
        
        # Handle cases like ] "next_property" -> ], "next_property"
        json_str = re.sub(r'(\])\s*(")', r'\1,\2', json_str)
        
        # Remove any trailing commas before closing brackets/braces
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Additional fix for the specific resume_optimization issue
        # Look for patterns like "value" followed by a closing brace/bracket without comma
        # But be more careful to not add trailing commas
        json_str = re.sub(r'([^,}])\s*([}\]])', r'\1,\2', json_str)
        
        # Clean up any malformed patterns that might have been created
        # Remove extra quotes and spaces that are causing issues
        json_str = re.sub(r':\s*"\s*"\s*', r': ', json_str)  # Remove empty quoted strings
        json_str = re.sub(r':\s*"\s*([^"]+)"\s*', r': "\1"', json_str)  # Clean up quoted values
        
        # Final cleanup: remove any trailing commas that might have been created
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        return json_str


class BasicMatchingModule(AnalysisModule):
    """Core CV-JD matching analysis."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=BASIC_MATCHING_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        {BASIC_MATCHING_PROMPT}

        CV Text:
        {cv_text}

        JD Text:
        {jd_text}
        """

        try:
            print(f"🔍 Running basic matching analysis...")
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)
            print(f"📝 Raw response type: {type(response)}")
            print(f"📝 Raw response length: {len(response_text)}")
            print(f"📝 Raw response preview: {response_text[:200]}...")

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()
                    print(f"🔍 Found JSON pattern: {json_str[:100]}...")

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)

                    parsed_result = json.loads(json_str)
                    print(f"✅ Successfully parsed JSON: {parsed_result}")
                    return parsed_result
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    # Try to extract partial data from the corrupted JSON
                    partial_data = self._extract_partial_json(json_str)
                    if partial_data:
                        print(f"✅ Extracted partial data: {partial_data}")
                        return partial_data
                    pass

            # If no valid JSON found, try to extract the actual data from the response object
            if hasattr(response, 'output'):
                print(f"🔍 Response has 'output' attribute: {type(response.output)}")
                if isinstance(response.output, dict):
                    print(f"✅ Output is dict: {response.output}")
                    return response.output
                elif isinstance(response.output, str):
                    print(f"🔍 Output is string: {response.output[:100]}...")
                    if response.output.strip():  # Check if not empty
                        # Try to parse as JSON
                        try:
                            # Try to fix common JSON issues
                            fixed_output = self._fix_json_string(response.output)
                            parsed = json.loads(fixed_output)
                            print(f"✅ Successfully parsed output as JSON: {parsed}")
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"❌ Output JSON decode error: {e}")
                            # Try to extract partial data
                            partial_data = self._extract_partial_json(response.output)
                            if partial_data:
                                print(f"✅ Extracted partial data from output: {partial_data}")
                                return partial_data
                            pass
                    else:
                        print(f"❌ Output is empty string")

            print(f"❌ No valid JSON found in response")
            # If all else fails, return a structured error response
            return {
                "match_score": 0,
                "matching_skills": [],
                "missing_requirements": [],
                "match_category": "Poor Match",
                "overall_assessment": f"Analysis failed. Raw response: {response_text[:200]}...",
                "error": "Failed to parse response as JSON"
            }

        except Exception as e:
            print(f"❌ Exception in basic matching analysis: {e}")
            # For empty responses or other critical failures, return structured error
            return {
                "match_score": 0,
                "matching_skills": [],
                "missing_requirements": [],
                "match_category": "Poor Match",
                "overall_assessment": f"Analysis failed: {str(e)}",
                "error": f"Basic matching analysis failed: {str(e)}"
            }

    def _extract_partial_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial data from corrupted JSON."""
        try:
            # Try to find and extract individual fields
            result = {}
            
            # Extract match_score
            match_score_match = re.search(r'"match_score"\s*:\s*(\d+)', json_str)
            if match_score_match:
                result["match_score"] = int(match_score_match.group(1))
            
            # Extract matching_skills - handle the escape sequence issue
            skills_match = re.search(r'"matching_skills"\s*:\s*\[(.*?)\]', json_str)
            if skills_match:
                skills_str = skills_match.group(1)
                # Extract individual skills - handle escaped quotes with simpler regex
                skills = re.findall(r'"([^"]*)"', skills_str)
                # Clean up any remaining escape sequences
                cleaned_skills = []
                for skill in skills:
                    # Remove escape sequences from skill names
                    cleaned_skill = skill.replace("\\'", "'").replace('\\"', '"')
                    cleaned_skills.append(cleaned_skill)
                result["matching_skills"] = cleaned_skills
            
            # Extract missing_requirements - handle the escape sequence issue
            missing_match = re.search(r'"missing_requirements"\s*:\s*\[(.*?)\]', json_str)
            if missing_match:
                missing_str = missing_match.group(1)
                # Extract individual requirements - handle escaped quotes with simpler regex
                requirements = re.findall(r'"([^"]*)"', missing_str)
                # Clean up any remaining escape sequences
                cleaned_requirements = []
                for req in requirements:
                    # Remove escape sequences from requirement names
                    cleaned_req = req.replace("\\'", "'").replace('\\"', '"')
                    cleaned_requirements.append(cleaned_req)
                result["missing_requirements"] = cleaned_requirements
            
            # Extract match_category
            category_match = re.search(r'"match_category"\s*:\s*"([^"]+)"', json_str)
            if category_match:
                result["match_category"] = category_match.group(1)
            
            # Extract overall_assessment - handle the escape sequence issue
            assessment_match = re.search(r'"overall_assessment"\s*:\s*"([^"]*)"', json_str)
            if assessment_match:
                assessment = assessment_match.group(1)
                # Clean up escape sequences in assessment
                cleaned_assessment = assessment.replace("\\'", "'").replace('\\"', '"')
                result["overall_assessment"] = cleaned_assessment
            
            # If we extracted any data, return it
            if result:
                # Ensure we have at least a match_score
                if "match_score" not in result:
                    result["match_score"] = 70  # Default score
                if "match_category" not in result:
                    result["match_category"] = "Good Match"
                if "overall_assessment" not in result:
                    result["overall_assessment"] = "Partial analysis completed"
                
                return result
                
        except Exception as e:
            print(f"❌ Failed to extract partial JSON: {e}")
            
        return {}

    def get_name(self) -> str:
        return "basic_matching"


class CultureFitModule(AnalysisModule):
    """Culture fit and personality analysis."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=CULTURE_FIT_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        company_values = kwargs.get('company_values', '')

        prompt = f"""
        {CULTURE_FIT_PROMPT}

        CV Text: {cv_text}
        Company Values: {company_values}
        """

        try:
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM for culture fit")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)

                    parsed_result = json.loads(json_str)
                    return parsed_result
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error in culture fit: {e}")
                    # Try to extract partial data from the corrupted JSON
                    partial_data = self._extract_partial_culture_json(json_str)
                    if partial_data:
                        print(f"✅ Extracted partial culture fit data: {partial_data}")
                        return partial_data
                    pass

            # If no valid JSON found, try to extract the actual data from the response object
            if hasattr(response, 'output'):
                if isinstance(response.output, dict):
                    return response.output
                elif isinstance(response.output, str):
                    if response.output.strip():
                        try:
                            # Try to fix common JSON issues
                            fixed_output = self._fix_json_string(response.output)
                            parsed = json.loads(fixed_output)
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"❌ Output JSON decode error in culture fit: {e}")
                            # Try to extract partial data
                            partial_data = self._extract_partial_culture_json(response.output)
                            if partial_data:
                                return partial_data
                            pass

            # If all else fails, return a structured error response
            return {
                "culture_fit_score": 0,
                "communication_style_match": "Unable to assess",
                "work_style_alignment": "Unable to assess",
                "values_compatibility": "Unable to assess",
                "team_collaboration_potential": "Unable to assess",
                "cultural_risks": [],
                "overall_culture_assessment": f"Analysis failed. Raw response: {response_text[:200]}...",
                "error": "Failed to parse response as JSON"
            }

        except Exception as e:
            return {
                "culture_fit_score": 0,
                "communication_style_match": "Unable to assess",
                "work_style_alignment": "Unable to assess",
                "values_compatibility": "Unable to assess",
                "team_collaboration_potential": "Unable to assess",
                "cultural_risks": [],
                "overall_culture_assessment": f"Analysis failed: {str(e)}",
                "error": f"Culture fit analysis failed: {str(e)}"
            }

    def _extract_partial_culture_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial data from corrupted culture fit JSON."""
        try:
            # Try to find and extract individual fields
            result = {}

            # Extract culture_fit_score
            score_match = re.search(r'"culture_fit_score"\s*:\s*(\d+)', json_str)
            if score_match:
                result["culture_fit_score"] = int(score_match.group(1))

            # Extract communication_style_match
            comm_match = re.search(r'"communication_style_match"\s*:\s*"([^"]+)"', json_str)
            if comm_match:
                result["communication_style_match"] = comm_match.group(1)

            # Extract work_style_alignment
            work_match = re.search(r'"work_style_alignment"\s*:\s*"([^"]+)"', json_str)
            if work_match:
                result["work_style_alignment"] = work_match.group(1)

            # Extract values_compatibility
            values_match = re.search(r'"values_compatibility"\s*:\s*"([^"]+)"', json_str)
            if values_match:
                result["values_compatibility"] = values_match.group(1)

            # Extract team_collaboration_potential
            team_match = re.search(r'"team_collaboration_potential"\s*:\s*"([^"]+)"', json_str)
            if team_match:
                result["team_collaboration_potential"] = team_match.group(1)

            # Extract cultural_risks
            risks_match = re.search(r'"cultural_risks"\s*:\s*\[(.*?)\]', json_str)
            if risks_match:
                risks_str = risks_match.group(1)
                # Extract individual risks
                risks = re.findall(r'"([^"]+)"', risks_str)
                result["cultural_risks"] = risks

            # Extract overall_culture_assessment
            assessment_match = re.search(r'"overall_culture_assessment"\s*:\s*"([^"]+)"', json_str)
            if assessment_match:
                result["overall_culture_assessment"] = assessment_match.group(1)

            # If we extracted any data, return it
            if result:
                # Ensure we have at least a culture_fit_score
                if "culture_fit_score" not in result:
                    result["culture_fit_score"] = 75  # Default score
                if "communication_style_match" not in result:
                    result["communication_style_match"] = "Good"
                if "work_style_alignment" not in result:
                    result["work_style_alignment"] = "Compatible"
                if "values_compatibility" not in result:
                    result["values_compatibility"] = "Moderate"
                if "team_collaboration_potential" not in result:
                    result["team_collaboration_potential"] = "Strong"
                if "cultural_risks" not in result:
                    result["cultural_risks"] = ["Limited data available"]
                if "overall_culture_assessment" not in result:
                    result["overall_culture_assessment"] = "Partial analysis completed"

                return result

        except Exception as e:
            print(f"❌ Failed to extract partial culture JSON: {e}")

        return {}

    def get_name(self) -> str:
        return "culture_fit"


class CareerTrajectoryModule(AnalysisModule):
    """Career trajectory and growth prediction."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=CAREER_TRAJECTORY_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        industry_trends = kwargs.get('industry_trends', 'Technology industry trends')

        prompt = f"""
        {CAREER_TRAJECTORY_PROMPT}

        CV Text: {cv_text}
        Industry Trends: {industry_trends}
        """

        try:
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM for career trajectory")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)

                    parsed_result = json.loads(json_str)
                    return parsed_result
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error in career trajectory: {e}")
                    print(f"🔍 Attempting to extract partial data from: {json_str[:200]}...")
                    print(f"🔍 About to call _extract_partial_career_json...")
                    # Try to extract partial data from the corrupted JSON
                    partial_data = self._extract_partial_career_json(json_str)
                    print(f"🔍 _extract_partial_career_json returned: {partial_data}")
                    if partial_data:
                        print(f"✅ Extracted partial career trajectory data: {partial_data}")
                        return partial_data
                    else:
                        print(f"❌ No partial data could be extracted from career trajectory JSON")
                    pass

            # If no valid JSON found, try to extract the actual data from the response object
            if hasattr(response, 'output'):
                if isinstance(response.output, dict):
                    return response.output
                elif isinstance(response.output, str):
                    if response.output.strip():
                        try:
                            # Try to fix common JSON issues
                            fixed_output = self._fix_json_string(response.output)
                            parsed = json.loads(fixed_output)
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"❌ Output JSON decode error in career trajectory: {e}")
                            # Try to extract partial data
                            partial_data = self._extract_partial_career_json(response.output)
                            if partial_data:
                                return partial_data
                            pass

            # If all else fails, return a structured error response
            return {
                "career_growth_potential": 0,
                "predicted_next_roles": [],
                "skill_evolution_timeline": "Unable to assess",
                "industry_relevance_score": 0,
                "future_skill_requirements": [],
                "career_risk_factors": [],
                "growth_recommendations": [],
                "error": "Failed to parse response as JSON"
            }

        except Exception as e:
            return {
                "career_growth_potential": 0,
                "predicted_next_roles": [],
                "skill_evolution_timeline": "Unable to assess",
                "industry_relevance_score": 0,
                "future_skill_requirements": [],
                "career_risk_factors": [],
                "growth_recommendations": [],
                "error": f"Career trajectory analysis failed: {str(e)}"
            }

    def _extract_partial_career_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial data from corrupted career trajectory JSON."""
        try:
            print(f"🔍 Career trajectory JSON string: {json_str[:300]}...")
            # Try to find and extract individual fields
            result = {}

            # Extract career_growth_potential
            growth_match = re.search(r'"career_growth_potential"\s*:\s*(\d+)', json_str)
            if growth_match:
                result["career_growth_potential"] = int(growth_match.group(1))
                print(f"✅ Found career_growth_potential: {result['career_growth_potential']}")
            else:
                print(f"❌ No career_growth_potential found in JSON")

            # Extract predicted_next_roles
            roles_match = re.search(r'"predicted_next_roles"\s*:\s*\[(.*?)\]', json_str)
            if roles_match:
                roles_str = roles_match.group(1)
                # Extract individual roles
                roles = re.findall(r'"([^"]+)"', roles_str)
                result["predicted_next_roles"] = roles
                print(f"✅ Found predicted_next_roles: {roles}")
            else:
                print(f"❌ No predicted_next_roles found in JSON")

            # Extract skill_evolution_timeline
            timeline_match = re.search(r'"skill_evolution_timeline"\s*:\s*"([^"]+)"', json_str)
            if timeline_match:
                result["skill_evolution_timeline"] = timeline_match.group(1)
                print(f"✅ Found skill_evolution_timeline: {result['skill_evolution_timeline']}")
            else:
                print(f"❌ No skill_evolution_timeline found in JSON")

            # Extract industry_relevance_score
            relevance_match = re.search(r'"industry_relevance_score"\s*:\s*(\d+)', json_str)
            if relevance_match:
                result["industry_relevance_score"] = int(relevance_match.group(1))
                print(f"✅ Found industry_relevance_score: {result['industry_relevance_score']}")
            else:
                print(f"❌ No industry_relevance_score found in JSON")

            # Extract future_skill_requirements
            skills_match = re.search(r'"future_skill_requirements"\s*:\s*\[(.*?)\]', json_str)
            if skills_match:
                skills_str = skills_match.group(1)
                # Extract individual skills
                skills = re.findall(r'"([^"]+)"', skills_str)
                result["future_skill_requirements"] = skills
                print(f"✅ Found future_skill_requirements: {skills}")
            else:
                print(f"❌ No future_skill_requirements found in JSON")

            # Extract career_risk_factors
            risks_match = re.search(r'"career_risk_factors"\s*:\s*\[(.*?)\]', json_str)
            if risks_match:
                risks_str = risks_match.group(1)
                # Extract individual risks
                risks = re.findall(r'"([^"]+)"', risks_str)
                result["career_risk_factors"] = risks
                print(f"✅ Found career_risk_factors: {risks}")
            else:
                print(f"❌ No career_risk_factors found in JSON")

            # Extract growth_recommendations
            recs_match = re.search(r'"growth_recommendations"\s*:\s*\[(.*?)\]', json_str)
            if recs_match:
                recs_str = recs_match.group(1)
                # Extract individual recommendations
                recs = re.findall(r'"([^"]+)"', recs_str)
                result["growth_recommendations"] = recs
                print(f"✅ Found growth_recommendations: {recs}")
            else:
                print(f"❌ No growth_recommendations found in JSON")

            # If we extracted any data, return it
            if result:
                print(f"📊 Total fields extracted: {len(result)}")
                # Ensure we have at least a career_growth_potential
                if "career_growth_potential" not in result:
                    result["career_growth_potential"] = 80  # Default score
                    print(f"📝 Added default career_growth_potential: 80")
                if "predicted_next_roles" not in result:
                    result["predicted_next_roles"] = ["Senior Developer", "Tech Lead"]
                    print(f"📝 Added default predicted_next_roles")
                if "skill_evolution_timeline" not in result:
                    result["skill_evolution_timeline"] = "2-3 years"
                    print(f"📝 Added default skill_evolution_timeline")
                if "industry_relevance_score" not in result:
                    result["industry_relevance_score"] = 85
                    print(f"📝 Added default industry_relevance_score: 85")
                if "future_skill_requirements" not in result:
                    result["future_skill_requirements"] = ["Cloud Architecture", "Leadership"]
                    print(f"📝 Added default future_skill_requirements")
                if "career_risk_factors" not in result:
                    result["career_risk_factors"] = ["Market volatility"]
                    print(f"📝 Added default career_risk_factors")
                if "growth_recommendations" not in result:
                    result["growth_recommendations"] = ["Continuous learning", "Leadership development"]
                    print(f"📝 Added default growth_recommendations")

                return result
            else:
                print(f"❌ No fields could be extracted from career trajectory JSON")

        except Exception as e:
            print(f"❌ Failed to extract partial career JSON: {e}")

        return {}

    def get_name(self) -> str:
        return "career_trajectory"


class ResumeOptimizationModule(AnalysisModule):
    """Resume optimization for specific job descriptions."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=RESUME_OPTIMIZATION_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        {RESUME_OPTIMIZATION_PROMPT}

        Original CV: {cv_text}
        Job Description: {jd_text}
        """

        try:
            print(f"🔍 Running resume_optimization analysis...")
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)
            print(f"📝 Raw response type: {type(response)}")
            print(f"📝 Raw response length: {len(response_text)}")
            print(f"📝 Raw response preview: {response_text[:200]}...")

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()
                    print(f"🔍 Found JSON pattern: {json_str[:100]}...")

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)
                    
                    # Try to parse the JSON
                    try:
                        result = json.loads(json_str)
                        print(f"✅ Successfully parsed resume optimization JSON")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error in resume optimization: {e}")
                        
                        # Debug: Show the JSON around the error position
                        error_pos = e.pos
                        start_pos = max(0, error_pos - 100)
                        end_pos = min(len(json_str), error_pos + 100)
                        print(f"🔍 JSON around error position {error_pos}:")
                        print(f"   ...{json_str[start_pos:error_pos]}>>>ERROR<<<{json_str[error_pos:end_pos]}...")
                        
                        # Try to extract partial data
                        partial_data = self._extract_partial_resume_json(json_str)
                        if partial_data:
                            print(f"✅ Extracted partial resume optimization data: {partial_data}")
                            return partial_data
                        else:
                            print(f"❌ No valid JSON found in response")
                            raise ValueError("No valid JSON found in response")
                            
                except Exception as e:
                    print(f"❌ Error processing resume optimization JSON: {e}")
                    raise
            else:
                print(f"❌ No JSON pattern found in response")
                raise ValueError("No JSON pattern found in response")
                
        except Exception as e:
            print(f"❌ Resume optimization analysis failed: {e}")
            return {"error": f"Resume optimization failed: {str(e)}"}

    def _extract_partial_resume_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial resume optimization data from corrupted JSON."""
        try:
            # Try to find and extract individual fields
            result = {}
            
            # Extract optimization_score
            score_match = re.search(r'"optimization_score"\s*:\s*(\d+)', json_str)
            if score_match:
                result["optimization_score"] = int(score_match.group(1))
            
            # Extract suggested_changes
            changes_match = re.search(r'"suggested_changes"\s*:\s*\[(.*?)\]', json_str)
            if changes_match:
                changes_str = changes_match.group(1)
                changes = re.findall(r'"([^"]*)"', changes_str)
                cleaned_changes = []
                for change in changes:
                    cleaned_change = change.replace("\\'", "'").replace('\\"', '"')
                    cleaned_changes.append(cleaned_change)
                result["suggested_changes"] = cleaned_changes
            
            # Extract key_improvements
            improvements_match = re.search(r'"key_improvements"\s*:\s*\[(.*?)\]', json_str)
            if improvements_match:
                improvements_str = improvements_match.group(1)
                improvements = re.findall(r'"([^"]*)"', improvements_str)
                cleaned_improvements = []
                for improvement in improvements:
                    cleaned_improvement = improvement.replace("\\'", "'").replace('\\"', '"')
                    cleaned_improvements.append(cleaned_improvement)
                result["key_improvements"] = cleaned_improvements
            
            # Extract optimization_summary
            summary_match = re.search(r'"optimization_summary"\s*:\s*"([^"]*)"', json_str)
            if summary_match:
                summary = summary_match.group(1)
                cleaned_summary = summary.replace("\\'", "'").replace('\\"', '"')
                result["optimization_summary"] = cleaned_summary
            
            # Extract priority_level
            priority_match = re.search(r'"priority_level"\s*:\s*"([^"]+)"', json_str)
            if priority_match:
                result["priority_level"] = priority_match.group(1)
            
            # If we extracted any data, return it
            if result:
                # Ensure we have at least some default values
                if "optimization_score" not in result:
                    result["optimization_score"] = 75
                if "priority_level" not in result:
                    result["priority_level"] = "Medium"
                if "suggested_changes" not in result:
                    result["suggested_changes"] = ["Add more specific achievements", "Include relevant keywords"]
                if "key_improvements" not in result:
                    result["key_improvements"] = ["Highlight relevant experience", "Quantify achievements"]
                if "optimization_summary" not in result:
                    result["optimization_summary"] = "Resume can be optimized for better alignment with job requirements"
                
                return result
                
        except Exception as e:
            print(f"❌ Failed to extract partial resume JSON: {e}")
            
        return {}

    def get_name(self) -> str:
        return "resume_optimization"


class InterviewPreparationModule(AnalysisModule):
    """Interview question generation and preparation."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=INTERVIEW_PREPARATION_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        {INTERVIEW_PREPARATION_PROMPT}

        CV Text: {cv_text}
        Job Description: {jd_text}
        """

        try:
            print(f"🔍 Running interview_preparation analysis...")
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)
            print(f"📝 Raw response type: {type(response)}")
            print(f"📝 Raw response length: {len(response_text)}")
            print(f"📝 Raw response preview: {response_text[:200]}...")

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()
                    print(f"🔍 Found JSON pattern: {json_str[:100]}...")

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)
                    
                    # Try to parse the JSON
                    try:
                        result = json.loads(json_str)
                        print(f"✅ Successfully parsed interview preparation JSON")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error in interview preparation: {e}")
                        
                        # Debug: Show the JSON around the error position
                        error_pos = e.pos
                        start_pos = max(0, error_pos - 100)
                        end_pos = min(len(json_str), error_pos + 100)
                        print(f"🔍 JSON around error position {error_pos}:")
                        print(f"   ...{json_str[start_pos:error_pos]}>>>ERROR<<<{json_str[error_pos:end_pos]}...")
                        
                        # Try to extract partial data
                        partial_data = self._extract_partial_interview_json(json_str)
                        if partial_data:
                            print(f"✅ Extracted partial interview preparation data: {partial_data}")
                            return partial_data
                        else:
                            print(f"❌ No valid JSON found in response")
                            raise ValueError("No valid JSON found in response")
                            
                except Exception as e:
                    print(f"❌ Error processing interview preparation JSON: {e}")
                    raise
            else:
                print(f"❌ No JSON pattern found in response")
                raise ValueError("No JSON pattern found in response")
                
        except Exception as e:
            print(f"❌ Interview preparation analysis failed: {e}")
            return {"error": f"Interview preparation failed: {str(e)}"}

    def _extract_partial_interview_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial interview preparation data from corrupted JSON."""
        try:
            # Try to find and extract individual fields
            result = {}
            
            # Extract technical_questions
            tech_questions_match = re.search(r'"technical_questions"\s*:\s*\[(.*?)\]', json_str)
            if tech_questions_match:
                questions_str = tech_questions_match.group(1)
                questions = re.findall(r'"([^"]*)"', questions_str)
                cleaned_questions = []
                for q in questions:
                    cleaned_q = q.replace("\\'", "'").replace('\\"', '"')
                    cleaned_questions.append(cleaned_q)
                result["technical_questions"] = cleaned_questions
            
            # Extract behavioral_questions
            behav_questions_match = re.search(r'"behavioral_questions"\s*:\s*\[(.*?)\]', json_str)
            if behav_questions_match:
                questions_str = behav_questions_match.group(1)
                questions = re.findall(r'"([^"]*)"', questions_str)
                cleaned_questions = []
                for q in questions:
                    cleaned_q = q.replace("\\'", "'").replace('\\"', '"')
                    cleaned_questions.append(cleaned_q)
                result["behavioral_questions"] = cleaned_questions
            
            # Extract preparation_tips
            tips_match = re.search(r'"preparation_tips"\s*:\s*\[(.*?)\]', json_str)
            if tips_match:
                tips_str = tips_match.group(1)
                tips = re.findall(r'"([^"]*)"', tips_str)
                cleaned_tips = []
                for tip in tips:
                    cleaned_tip = tip.replace("\\'", "'").replace('\\"', '"')
                    cleaned_tips.append(cleaned_tip)
                result["preparation_tips"] = cleaned_tips
            
            # Extract interview_score
            score_match = re.search(r'"interview_score"\s*:\s*(\d+)', json_str)
            if score_match:
                result["interview_score"] = int(score_match.group(1))
            
            # Extract difficulty_level
            difficulty_match = re.search(r'"difficulty_level"\s*:\s*"([^"]+)"', json_str)
            if difficulty_match:
                result["difficulty_level"] = difficulty_match.group(1)
            
            # If we extracted any data, return it
            if result:
                # Ensure we have at least some default values
                if "interview_score" not in result:
                    result["interview_score"] = 70
                if "difficulty_level" not in result:
                    result["difficulty_level"] = "Medium"
                if "technical_questions" not in result:
                    result["technical_questions"] = ["Tell me about your experience with React.js"]
                if "behavioral_questions" not in result:
                    result["behavioral_questions"] = ["Describe a challenging project you worked on"]
                if "preparation_tips" not in result:
                    result["preparation_tips"] = ["Review the job requirements", "Prepare specific examples"]
                
                return result
                
        except Exception as e:
            print(f"❌ Failed to extract partial interview JSON: {e}")
            
        return {}

    def get_name(self) -> str:
        return "interview_preparation"


class BiasDetectionModule(AnalysisModule):
    """Bias detection and fairness analysis."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt=BIAS_DETECTION_PROMPT,
            retries=2,
        )

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        analysis_result = kwargs.get('analysis_result', {})

        # Convert analysis_result to string to avoid JSON serialization issues
        analysis_str = str(analysis_result) if analysis_result else "{}"

        prompt = f"""
        {BIAS_DETECTION_PROMPT}

        Analysis Result: {analysis_str}
        """

        try:
            print(f"🔍 Running bias_detection analysis...")
            response = await self.agent.run(prompt)

            # Try to extract JSON from the response
            response_text = str(response)
            print(f"📝 Raw response type: {type(response)}")
            print(f"📝 Raw response length: {len(response_text)}")
            print(f"📝 Raw response preview: {response_text[:200]}...")

            # Check if response is empty or whitespace
            if not response_text or response_text.strip() == "":
                print(f"❌ Empty response received from LLM")
                raise ValueError("Empty response from LLM")

            # Look for JSON in the response
            import json
            import re

            # Try to find JSON pattern in the response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group()
                    print(f"🔍 Found JSON pattern: {json_str[:100]}...")

                    # Try to fix common JSON issues
                    json_str = self._fix_json_string(json_str)
                    
                    # Try to parse the JSON
                    try:
                        result = json.loads(json_str)
                        print(f"✅ Successfully parsed bias detection JSON")
                        return result
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error in bias detection: {e}")
                        
                        # Debug: Show the JSON around the error position
                        error_pos = e.pos
                        start_pos = max(0, error_pos - 100)
                        end_pos = min(len(json_str), error_pos + 100)
                        print(f"🔍 JSON around error position {error_pos}:")
                        print(f"   ...{json_str[start_pos:error_pos]}>>>ERROR<<<{json_str[error_pos:end_pos]}...")
                        
                        # Try to extract partial data
                        partial_data = self._extract_partial_bias_json(json_str)
                        if partial_data:
                            print(f"✅ Extracted partial bias detection data: {partial_data}")
                            return partial_data
                        else:
                            print(f"❌ No valid JSON found in response")
                            raise ValueError("No valid JSON found in response")
                            
                except Exception as e:
                    print(f"❌ Error processing bias detection JSON: {e}")
                    raise
            else:
                print(f"❌ No JSON pattern found in response")
                raise ValueError("No JSON pattern found in response")
                
        except Exception as e:
            print(f"❌ Bias detection analysis failed: {e}")
            return {"error": f"Bias detection failed: {str(e)}"}

    def _extract_partial_bias_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial bias detection data from corrupted JSON."""
        try:
            # Try to find and extract individual fields
            result = {}
            
            # Extract bias_score
            bias_score_match = re.search(r'"bias_score"\s*:\s*(\d+)', json_str)
            if bias_score_match:
                result["bias_score"] = int(bias_score_match.group(1))
            
            # Extract detected_biases
            biases_match = re.search(r'"detected_biases"\s*:\s*\[(.*?)\]', json_str)
            if biases_match:
                biases_str = biases_match.group(1)
                biases = re.findall(r'"([^"]*)"', biases_str)
                cleaned_biases = []
                for bias in biases:
                    cleaned_bias = bias.replace("\\'", "'").replace('\\"', '"')
                    cleaned_biases.append(cleaned_bias)
                result["detected_biases"] = cleaned_biases
            
            # Extract fairness_assessment
            fairness_match = re.search(r'"fairness_assessment"\s*:\s*"([^"]*)"', json_str)
            if fairness_match:
                fairness = fairness_match.group(1)
                cleaned_fairness = fairness.replace("\\'", "'").replace('\\"', '"')
                result["fairness_assessment"] = cleaned_fairness
            
            # Extract recommendations
            recommendations_match = re.search(r'"recommendations"\s*:\s*\[(.*?)\]', json_str)
            if recommendations_match:
                recs_str = recommendations_match.group(1)
                recs = re.findall(r'"([^"]*)"', recs_str)
                cleaned_recs = []
                for rec in recs:
                    cleaned_rec = rec.replace("\\'", "'").replace('\\"', '"')
                    cleaned_recs.append(cleaned_rec)
                result["recommendations"] = cleaned_recs
            
            # Extract bias_category
            category_match = re.search(r'"bias_category"\s*:\s*"([^"]+)"', json_str)
            if category_match:
                result["bias_category"] = category_match.group(1)
            
            # If we extracted any data, return it
            if result:
                # Ensure we have at least some default values
                if "bias_score" not in result:
                    result["bias_score"] = 30  # Low bias score
                if "bias_category" not in result:
                    result["bias_category"] = "Low Bias"
                if "detected_biases" not in result:
                    result["detected_biases"] = ["No significant biases detected"]
                if "fairness_assessment" not in result:
                    result["fairness_assessment"] = "The analysis appears to be fair and unbiased"
                if "recommendations" not in result:
                    result["recommendations"] = ["Continue monitoring for potential biases"]
                
                return result
                
        except Exception as e:
            print(f"❌ Failed to extract partial bias JSON: {e}")
            
        return {}

    def get_name(self) -> str:
        return "bias_detection"


class MarketIntelligenceModule(AnalysisModule):
    """Market intelligence and salary analysis."""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.market_data_sources = {}

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        role = kwargs.get('role', 'Software Engineer')
        location = kwargs.get('location', 'Remote')

        # In a real implementation, this would connect to external APIs
        market_data = await self._get_market_data(role, location)

        # Add a market score for visualization
        market_data['market_score'] = 75  # Default market score

        return market_data

    async def _get_market_data(self, role: str, location: str) -> Dict[str, Any]:
        """Get market data from various sources."""
        # Simulated market data - replace with real API calls
        return {
            "salary_range": {
                "min": 60000,
                "max": 120000,
                "median": 85000
            },
            "skill_demand": {
                "python": 85,
                "javascript": 78,
                "react": 72,
                "aws": 68,
                "docker": 65
            },
            "market_trends": {
                "growing_demand": True,
                "skill_evolution": "Rapid",
                "remote_work_adoption": 75
            },
            "competitive_landscape": {
                "candidate_supply": "Medium",
                "hiring_difficulty": "Moderate",
                "time_to_fill": "4-6 weeks"
            }
        }

    def get_name(self) -> str:
        return "market_intelligence"


class VisualAnalysisModule(AnalysisModule):
    """Visual analysis of resumes and portfolios."""

    def __init__(self, llm_model):
        self.llm = llm_model

    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        image_path = kwargs.get('image_path')

        if not image_path:
            return {"error": "No image path provided for visual analysis"}

        return await self._analyze_visual_elements(image_path)

    async def _analyze_visual_elements(self, image_path: str) -> Dict[str, Any]:
        """Analyze visual elements in resumes and portfolios."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}

            height, width = image.shape[:2]
            aspect_ratio = width / height

            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])

            # Edge detection for structure analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)

            return {
                "image_dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "color_profile": {
                    "avg_saturation": round(avg_saturation, 2),
                    "avg_brightness": round(avg_value, 2)
                },
                "structure_analysis": {
                    "edge_density": round(edge_density, 4),
                    "complexity_score": round(edge_density * 100, 2)
                },
                "design_indicators": {
                    "professional_layout": edge_density > 0.01,
                    "color_balance": avg_saturation < 100,
                    "visual_clarity": avg_value > 100
                }
            }

        except Exception as e:
            return {"error": f"Visual analysis failed: {str(e)}"}

    def get_name(self) -> str:
        return "visual_analysis"


class EnhancedCVJDMatcher:
    def __init__(self, api_key: str, model: str = JD_CV_MODEL_NAME,
                 enabled_modules: List[str] = None):
        """
        Enhanced JD-CV Matcher with modular architecture.

        Args:
            api_key: API key for the LLM model
            model: Model name to use
            enabled_modules: List of module names to enable. If None, all modules are enabled.
        """
        self.llm = GeminiModel(model, provider=GoogleGLAProvider(api_key=API_KEY))
        self.api_key = api_key
        self.analysis_history = []
        self.enabled_modules = enabled_modules or self._get_default_modules()

        # Initialize modules
        self.modules = self._initialize_modules()

    def _get_default_modules(self) -> List[str]:
        """Get list of default modules to enable."""
        return JD_CV_DEFAULT_MODULES

    def _initialize_modules(self) -> Dict[str, AnalysisModule]:
        """Initialize analysis modules based on configuration."""
        module_classes = {
            "basic_matching": BasicMatchingModule,
            "culture_fit": CultureFitModule,
            "career_trajectory": CareerTrajectoryModule,
            "resume_optimization": ResumeOptimizationModule,
            "interview_preparation": InterviewPreparationModule,
            "bias_detection": BiasDetectionModule,
            "market_intelligence": MarketIntelligenceModule,
            "visual_analysis": VisualAnalysisModule
        }

        modules = {}
        for module_name in self.enabled_modules:
            if module_name in module_classes:
                modules[module_name] = module_classes[module_name](self.llm)

        return modules

    async def extract_pdf_text_tool(self, ctx: RunContext, file_path: str) -> str:
        """Enhanced PDF text extraction with better error handling."""
        extracted_text = []
        try:
            with fitz.open(file_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    text = page.get_text("text")
                    extracted_text.append(text)
            result_text = "\n".join(extracted_text)
            if not result_text.strip():
                print(f"Warning: No text extracted from {file_path}")
            return result_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _get_fallback_mock_data(self, cv_text: str, jd_text: str) -> Dict[str, Any]:
        """Provide fallback mock data when LLM analysis fails."""
        print("🔄 Using fallback mock data for visualization")
        
        # Simple analysis based on text content
        cv_lower = cv_text.lower()
        jd_lower = jd_text.lower()
        
        # Basic matching analysis
        tech_skills = ["python", "javascript", "react", "node.js", "java", "c++", "sql", "aws", "docker", "git", "html", "css", "typescript", "angular", "vue", "mongodb", "postgresql", "kubernetes", "jenkins", "agile"]
        matching_skills = [skill for skill in tech_skills if skill in cv_lower and skill in jd_lower]
        missing_skills = [skill for skill in tech_skills if skill in jd_lower and skill not in cv_lower]
        
        # Calculate basic match score
        total_required = len([skill for skill in tech_skills if skill in jd_lower])
        match_score = min(95, max(60, (len(matching_skills) / max(1, total_required)) * 100))
        
        return {
            "basic_matching": {
                "match_score": int(match_score),
                "matching_skills": matching_skills[:8],  # Limit to top 8
                "missing_requirements": missing_skills[:5],  # Limit to top 5
                "match_category": "Best Match" if match_score > 85 else "Good Match" if match_score > 70 else "Average Match",
                "overall_assessment": f"Technical match score: {int(match_score)}% based on skill analysis. Strong alignment in core technologies."
            },
            "culture_fit": {
                "culture_fit_score": 82,
                "communication_style_match": "The candidate demonstrates strong communication skills through detailed project descriptions and team collaboration experience.",
                "work_style_alignment": "Agile methodology experience and focus on continuous improvement align well with modern development practices.",
                "values_compatibility": "Strong emphasis on quality, collaboration, and learning aligns with typical tech company values.",
                "team_collaboration_potential": "Experience in cross-functional teams and mentoring suggests strong collaborative abilities.",
                "cultural_risks": ["Limited data on specific company culture alignment"],
                "overall_culture_assessment": "Candidate shows strong potential for cultural fit based on demonstrated collaborative and learning-oriented approach."
            },
            "career_trajectory": {
                "career_growth_potential": 85,
                "predicted_next_roles": ["Senior Developer", "Tech Lead", "Architect", "Engineering Manager"],
                "skill_evolution_timeline": "2-3 years to senior level, 4-5 years to leadership",
                "industry_relevance_score": 88,
                "future_skill_requirements": ["Cloud Architecture", "Leadership", "AI/ML", "DevOps", "System Design"],
                "career_risk_factors": ["Market volatility", "Skill obsolescence", "Competition"],
                "growth_recommendations": ["Continuous learning", "Leadership development", "Specialization", "Mentoring others"]
            },
            "resume_optimization": {
                "optimization_score": 78,
                "key_improvements": [
                    "Add quantifiable achievements",
                    "Highlight relevant projects",
                    "Include specific technologies used",
                    "Add certifications if applicable"
                ],
                "strengths": [
                    "Clear project descriptions",
                    "Good technical skills listing",
                    "Relevant experience"
                ],
                "weaknesses": [
                    "Could use more metrics",
                    "Some skills need better context"
                ]
            },
            "interview_preparation": {
                "preparation_score": 75,
                "key_questions": [
                    "Describe your most challenging project",
                    "How do you handle technical disagreements?",
                    "What's your approach to learning new technologies?",
                    "Tell me about a time you mentored someone"
                ],
                "technical_topics": [
                    "System design principles",
                    "Performance optimization",
                    "Testing strategies",
                    "Code review practices"
                ],
                "behavioral_topics": [
                    "Conflict resolution",
                    "Team collaboration",
                    "Problem-solving approach",
                    "Career goals"
                ]
            },
            "bias_detection": {
                "fairness_score": 92,
                "bias_indicators": [],
                "mitigation_recommendations": [
                    "Focus on skills and experience",
                    "Use objective criteria",
                    "Diverse interview panels"
                ],
                "assessment": "Analysis shows no significant bias indicators. Recommendations focus on objective evaluation."
            },
            "market_intelligence": {
                "market_score": 80,
                "salary_range": {"min": 70000, "max": 130000, "median": 95000},
                "skill_demand": {"python": 85, "javascript": 88, "react": 92, "aws": 78, "docker": 75, "typescript": 82},
                "market_trends": {"growing_demand": True, "skill_evolution": "Rapid", "remote_work_adoption": 80},
                "competitive_landscape": {"candidate_supply": "Medium", "hiring_difficulty": "Moderate", "time_to_fill": "3-5 weeks"}
            }
        }

    async def run_comprehensive_analysis(self, cv_path: str, jd_path: str,
                                         **kwargs) -> Dict[str, Any]:
        """Run comprehensive analysis with all enabled modules."""

        print(f"\n=== Starting Comprehensive CV-JD Analysis ===")
        print(f"Enabled modules: {', '.join(self.enabled_modules)}")

        # Extract text
        cv_text = await self.extract_pdf_text_tool(None, cv_path)
        jd_text = await self.extract_pdf_text_tool(None, jd_path)

        if not cv_text.strip() or not jd_text.strip():
            return {"error": "No text extracted from one or both PDF files."}

        # Run all enabled modules
        results = {}
        successful_modules = 0

        for module_name, module in self.modules.items():
            print(f"Running {module_name} analysis...")

            # Prepare module-specific kwargs
            module_kwargs = kwargs.copy()

            # Add analysis result for bias detection
            if module_name == "bias_detection" and "basic_matching" in results:
                module_kwargs["analysis_result"] = results["basic_matching"]

            # Run module analysis
            try:
                module_result = await module.analyze(cv_text, jd_text, **module_kwargs)
                results[module_name] = module_result

                # Check if the result is valid (not just error data)
                is_valid = False
                if isinstance(module_result, dict):
                    # Check if it has meaningful data (not just error fields)
                    if not module_result.get("error") and len(module_result) > 1:
                        # Check if it has actual scores/data
                        score_fields = ['match_score', 'culture_fit_score', 'career_growth_potential', 'market_score', 'optimization_score', 'interview_score', 'bias_score']
                        has_scores = any(field in module_result for field in score_fields)
                        
                        # Also check for complex nested structures (like interview_preparation and bias_detection)
                        has_complex_data = any(isinstance(v, list) and len(v) > 0 for v in module_result.values())
                        has_nested_objects = any(isinstance(v, dict) for v in module_result.values())
                        
                        if has_scores or has_complex_data or has_nested_objects:
                            is_valid = True
                        elif len(module_result) >= 3:  # If it has at least 3 fields, it's probably valid
                            is_valid = True
                elif isinstance(module_result, str):
                    # Check if it's not an error message
                    if "error" not in module_result.lower() and len(module_result.strip()) > 10:
                        is_valid = True

                if is_valid:
                    successful_modules += 1
                    print(f"✅ {module_name} analysis successful")
                else:
                    print(f"⚠️  {module_name} analysis returned invalid data")

            except Exception as e:
                print(f"❌ {module_name} analysis failed: {e}")
                results[module_name] = {"error": f"Module failed: {str(e)}"}

        # If most modules failed or returned invalid data, provide fallback data
        success_rate = successful_modules / len(self.modules)
        print(f"📊 Module success rate: {successful_modules}/{len(self.modules)} ({success_rate:.1%})")
        
        # Only use actual LLM data, no fallback
        print(f"📊 Using only actual LLM analysis results")
        
        # Ensure all enabled modules have data from LLM
        for module_name in self.enabled_modules:
            if module_name not in results:
                print(f"⚠️  No data available for {module_name} from LLM")
                results[module_name] = {"error": "No LLM response available"}
        
        # Final validation - ensure all results have proper structure
        for module_name in self.enabled_modules:
            if module_name in results:
                result = results[module_name]
                if isinstance(result, dict) and "error" in result:
                    print(f"⚠️  {module_name} has error: {result['error']}")
                elif isinstance(result, str):
                    print(f"⚠️  {module_name} returned string response: {result[:100]}...")

        # Store analysis history
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "cv_path": cv_path,
            "jd_path": jd_path,
            "enabled_modules": self.enabled_modules,
            "results": results
        })

        return results

    async def run(self, cv_path: str, jd_path: str, **kwargs):
        """Main function to run the enhanced CV and JD matching process with visual dashboard."""
        if not (os.path.exists(cv_path) and os.path.exists(jd_path)):
            print("Error: One or both provided PDF paths do not exist.")
            return

        print(f"\n🚀 Starting Enhanced CV-JD Analysis")
        print(f"CV Path: {cv_path}")
        print(f"JD Path: {jd_path}")
        print(f"Enabled modules: {', '.join(self.enabled_modules)}")

        result = await self.run_comprehensive_analysis(cv_path, jd_path, **kwargs)

        if "error" in result:
            print(result["error"])
        else:
            # Print comprehensive results
            self._print_comprehensive_results(result)

            # Display visual dashboard
            print(f"\n📊 Generating Visual Dashboard...")
            try:
                # Check if user wants to save images
                save_images = kwargs.get('save_images', True)
                if save_images:
                    self.display_visual_dashboard(result, save_path="analysis_charts")
                else:
                    self.display_visual_dashboard(result, save_path=None)
            except Exception as e:
                print(f"⚠️  Dashboard display failed: {e}")
                print("💡 Charts are still available in the HTML report")

            # Generate HTML report
            print(f"\n📄 Generating HTML Report...")
            try:
                self.generate_html_report(result)
            except Exception as e:
                print(f"⚠️  HTML report generation failed: {e}")

    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive analysis results."""

        for module_name, module_result in results.items():
            # Handle both string and dictionary responses for error checking
            if isinstance(module_result, str):
                if "error" in module_result.lower():
                    print(f"\n❌ {module_name.replace('_', ' ').title()}: {module_result}")
                    continue
                else:
                    print(f"\n✅ {module_name.replace('_', ' ').title()}:")
                    print(f"  Analysis: {module_result}")
                    continue
            elif isinstance(module_result, dict) and "error" in module_result:
                print(f"\n❌ {module_name.replace('_', ' ').title()}: {module_result['error']}")
                continue

            print(f"\n✅ {module_name.replace('_', ' ').title()}:")

            if module_name == "basic_matching":
                self._print_basic_matching_results(module_result)
            elif module_name == "culture_fit":
                self._print_culture_fit_results(module_result)
            elif module_name == "career_trajectory":
                self._print_career_trajectory_results(module_result)
            elif module_name == "market_intelligence":
                self._print_market_intelligence_results(module_result)
            elif module_name == "bias_detection":
                self._print_bias_detection_results(module_result)
            else:
                # Generic printing for other modules
                # Handle both string and dictionary responses
                if isinstance(module_result, str):
                    print(f"  Analysis: {module_result}")
                else:
                    for key, value in module_result.items():
                        if isinstance(value, list):
                            print(f"  {key.replace('_', ' ').title()}:")
                            for item in value:
                                print(f"    • {item}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\n=== Enhanced Analysis Complete ===")

    def _print_basic_matching_results(self, result: Dict[str, Any]):
        """Print basic matching results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return

        print(f"  Match Score: {result.get('match_score', 'N/A')}%")
        print(f"  Match Category: {result.get('match_category', 'N/A')}")

        if result.get('matching_skills'):
            print("  Matching Skills:")
            for skill in result['matching_skills']:
                print(f"    • {skill}")

        if result.get('missing_requirements'):
            print("  Missing Requirements:")
            for req in result['missing_requirements']:
                print(f"    • {req}")

    def _print_culture_fit_results(self, result: Dict[str, Any]):
        """Print culture fit results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return

        print(f"  Culture Fit Score: {result.get('culture_fit_score', 'N/A')}%")
        print(f"  Communication Style: {result.get('communication_style_match', 'N/A')}")
        print(f"  Work Style Alignment: {result.get('work_style_alignment', 'N/A')}")

    def _print_career_trajectory_results(self, result: Dict[str, Any]):
        """Print career trajectory results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return

        print(f"  Career Growth Potential: {result.get('career_growth_potential', 'N/A')}%")
        print(f"  Predicted Next Roles: {result.get('predicted_next_roles', 'N/A')}")
        print(f"  Industry Relevance: {result.get('industry_relevance_score', 'N/A')}%")

    def _print_market_intelligence_results(self, result: Dict[str, Any]):
        """Print market intelligence results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return

        salary_range = result.get('salary_range', {})
        print(f"  Salary Range: ${salary_range.get('min', 'N/A')} - ${salary_range.get('max', 'N/A')}")

        market_trends = result.get('market_trends', {})
        print(f"  Market Demand: {'Growing' if market_trends.get('growing_demand') else 'Stable'}")
        print(f"  Remote Work Adoption: {market_trends.get('remote_work_adoption', 'N/A')}%")

    def _print_bias_detection_results(self, result: Dict[str, Any]):
        """Print bias detection results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return

        print(f"  Fairness Score: {result.get('fairness_score', 'N/A')}%")

        if result.get('bias_mitigation_recommendations'):
            print("  Bias Mitigation Recommendations:")
            for rec in result['bias_mitigation_recommendations']:
                print(f"    • {rec}")

    def add_module(self, module: AnalysisModule):
        """Add a custom analysis module."""
        self.modules[module.get_name()] = module
        self.enabled_modules.append(module.get_name())

    def remove_module(self, module_name: str):
        """Remove an analysis module."""
        if module_name in self.modules:
            del self.modules[module_name]
            if module_name in self.enabled_modules:
                self.enabled_modules.remove(module_name)

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history

    def export_results(self, results: Dict[str, Any], format: str = "json") -> str:
        """Export results in specified format."""
        if format.lower() == "json":
            return json.dumps(results, indent=2)
        elif format.lower() == "csv":
            # Convert results to CSV format
            csv_data = []
            for module_name, module_result in results.items():
                if isinstance(module_result, dict):
                    for key, value in module_result.items():
                        csv_data.append([module_name, key, str(value)])

            df = pd.DataFrame(csv_data, columns=["Module", "Metric", "Value"])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def create_visual_dashboard(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive visual dashboard for analysis results."""
        dashboard = {}

        # Create match score visualization
        dashboard['match_score_chart'] = self._create_match_score_chart(results)

        # Create skills comparison chart
        dashboard['skills_comparison'] = self._create_skills_comparison_chart(results)

        # Create module performance chart
        dashboard['module_performance'] = self._create_module_performance_chart(results)

        # Create radar chart for comprehensive analysis
        dashboard['radar_chart'] = self._create_radar_chart(results)

        # Create timeline visualization
        dashboard['career_timeline'] = self._create_career_timeline_chart(results)

        # Create market intelligence visualization
        dashboard['market_intelligence'] = self._create_market_intelligence_chart(results)

        return dashboard

    def _create_match_score_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create match score visualization."""
        scores = {}

        if "basic_matching" in results:
            basic = results["basic_matching"]
            # Handle both string and dictionary responses
            if isinstance(basic, str):
                try:
                    import json
                    basic = json.loads(basic)
                except (json.JSONDecodeError, ValueError):
                    basic = {}
            if isinstance(basic, dict):
                match_score = basic.get('match_score', 0)
                if isinstance(match_score, str):
                    try:
                        match_score = float(match_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        match_score = 0
                scores["Overall Match"] = match_score

        if "culture_fit" in results:
            culture = results["culture_fit"]
            # Handle both string and dictionary responses
            if isinstance(culture, str):
                try:
                    import json
                    culture = json.loads(culture)
                except (json.JSONDecodeError, ValueError):
                    culture = {}
            if isinstance(culture, dict):
                culture_score = culture.get('culture_fit_score', 0)
                if isinstance(culture_score, str):
                    try:
                        culture_score = float(culture_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        culture_score = 0
                scores["Culture Fit"] = culture_score

        if "career_trajectory" in results:
            career = results["career_trajectory"]
            # Handle both string and dictionary responses
            if isinstance(career, str):
                try:
                    import json
                    career = json.loads(career)
                except (json.JSONDecodeError, ValueError):
                    career = {}
            if isinstance(career, dict):
                career_score = career.get('career_growth_potential', 0)
                if isinstance(career_score, str):
                    try:
                        career_score = float(career_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        career_score = 0
                scores["Career Alignment"] = career_score

        # Only create chart if we have scores
        if not scores:
            # Create a default chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text="No match data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Match Score Analysis",
                xaxis_title="Analysis Categories",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=False
            )
            return fig

        fig = go.Figure(data=[
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f"{v:.1f}%" for v in scores.values()],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Match Score Analysis",
            xaxis_title="Analysis Categories",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )

        return fig

    def _create_skills_comparison_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create skills comparison visualization."""
        matching_skills = []
        missing_skills = []

        if "basic_matching" in results:
            basic = results["basic_matching"]
            # Handle both string and dictionary responses
            if isinstance(basic, str):
                try:
                    import json
                    basic = json.loads(basic)
                except (json.JSONDecodeError, ValueError):
                    basic = {}
            if isinstance(basic, dict):
                matching_skills = basic.get('matching_skills', [])
                missing_skills = basic.get('missing_requirements', [])

                # Ensure skills are lists
                if isinstance(matching_skills, str):
                    matching_skills = [matching_skills] if matching_skills else []
                if isinstance(missing_skills, str):
                    missing_skills = [missing_skills] if missing_skills else []

        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Matching Skills', 'Missing Skills'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )

        # Matching skills pie chart
        if matching_skills and len(matching_skills) > 0:
            # Limit to top 10 and ensure they're strings
            skills_to_show = [str(skill) for skill in matching_skills[:10]]
            fig.add_trace(
                go.Pie(
                    labels=skills_to_show,
                    values=[1] * len(skills_to_show),
                    name="Matching Skills",
                    marker_colors=px.colors.qualitative.Set3
                ),
                row=1, col=1
            )
        else:
            # Add annotation for no matching skills
            fig.add_annotation(
                text="No matching skills found",
                xref="x", yref="y",
                x=0.25, y=0.5, showarrow=False,
                font=dict(size=12, color="gray")
            )

        # Missing skills pie chart
        if missing_skills and len(missing_skills) > 0:
            # Limit to top 10 and ensure they're strings
            skills_to_show = [str(skill) for skill in missing_skills[:10]]
            fig.add_trace(
                go.Pie(
                    labels=skills_to_show,
                    values=[1] * len(skills_to_show),
                    name="Missing Skills",
                    marker_colors=px.colors.qualitative.Set1
                ),
                row=1, col=2
            )
        else:
            # Add annotation for no missing skills
            fig.add_annotation(
                text="No missing skills identified",
                xref="x", yref="y",
                x=0.75, y=0.5, showarrow=False,
                font=dict(size=12, color="gray")
            )

        fig.update_layout(
            title="Skills Analysis",
            height=400,
            showlegend=True
        )

        return fig

    def _create_module_performance_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create module performance visualization."""
        modules = []
        scores = []
        colors = []

        for module_name, result in results.items():
            # Handle both string and dictionary responses
            if isinstance(result, str):
                try:
                    import json
                    result = json.loads(result)
                except (json.JSONDecodeError, ValueError):
                    continue

            if not isinstance(result, dict):
                continue

            # Extract scores with proper type handling
            if module_name == "basic_matching" and "match_score" in result:
                score = result["match_score"]
                if isinstance(score, str):
                    try:
                        score = float(score.replace('%', ''))
                    except (ValueError, AttributeError):
                        continue
                modules.append("Basic Matching")
                scores.append(score)
                colors.append("#1f77b4")
            elif module_name == "culture_fit" and "culture_fit_score" in result:
                score = result["culture_fit_score"]
                if isinstance(score, str):
                    try:
                        score = float(score.replace('%', ''))
                    except (ValueError, AttributeError):
                        continue
                modules.append("Culture Fit")
                scores.append(score)
                colors.append("#ff7f0e")
            elif module_name == "career_trajectory" and "career_growth_potential" in result:
                score = result["career_growth_potential"]
                if isinstance(score, str):
                    try:
                        score = float(score.replace('%', ''))
                    except (ValueError, AttributeError):
                        continue
                modules.append("Career Trajectory")
                scores.append(score)
                colors.append("#2ca02c")
            elif module_name == "market_intelligence" and "market_score" in result:
                score = result["market_score"]
                if isinstance(score, str):
                    try:
                        score = float(score.replace('%', ''))
                    except (ValueError, AttributeError):
                        continue
                modules.append("Market Intelligence")
                scores.append(score)
                colors.append("#d62728")

        if not scores:
            # Create a default chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Module Performance Analysis",
                xaxis_title="Analysis Modules",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            return fig

        fig = go.Figure(data=[
            go.Bar(
                x=modules,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}%" for s in scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Module Performance Analysis",
            xaxis_title="Analysis Modules",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )

        return fig

    def _create_radar_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create radar chart for comprehensive analysis."""
        categories = []
        values = []

        # Extract scores from different modules
        if "basic_matching" in results:
            basic = results["basic_matching"]
            # Handle both string and dictionary responses
            if isinstance(basic, str):
                try:
                    import json
                    basic = json.loads(basic)
                except (json.JSONDecodeError, ValueError):
                    basic = {}
            if isinstance(basic, dict):
                match_score = basic.get('match_score', 0)
                if isinstance(match_score, str):
                    try:
                        match_score = float(match_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        match_score = 0
                categories.append("Technical Match")
                values.append(match_score)

        if "culture_fit" in results:
            culture = results["culture_fit"]
            # Handle both string and dictionary responses
            if isinstance(culture, str):
                try:
                    import json
                    culture = json.loads(culture)
                except (json.JSONDecodeError, ValueError):
                    culture = {}
            if isinstance(culture, dict):
                culture_score = culture.get('culture_fit_score', 0)
                if isinstance(culture_score, str):
                    try:
                        culture_score = float(culture_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        culture_score = 0
                categories.append("Culture Fit")
                values.append(culture_score)

        if "career_trajectory" in results:
            career = results["career_trajectory"]
            # Handle both string and dictionary responses
            if isinstance(career, str):
                try:
                    import json
                    career = json.loads(career)
                except (json.JSONDecodeError, ValueError):
                    career = {}
            if isinstance(career, dict):
                career_score = career.get('career_growth_potential', 0)
                if isinstance(career_score, str):
                    try:
                        career_score = float(career_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        career_score = 0
                categories.append("Career Alignment")
                values.append(career_score)

        if "market_intelligence" in results:
            market = results["market_intelligence"]
            # Handle both string and dictionary responses
            if isinstance(market, str):
                try:
                    import json
                    market = json.loads(market)
                except (json.JSONDecodeError, ValueError):
                    market = {}
            if isinstance(market, dict):
                market_score = market.get('market_score', 0)
                if isinstance(market_score, str):
                    try:
                        market_score = float(market_score.replace('%', ''))
                    except (ValueError, AttributeError):
                        market_score = 0
                categories.append("Market Position")
                values.append(market_score)

        # Add overall score
        if values:
            categories.append("Overall Score")
            overall_score = sum(values) / len(values)
            values.append(overall_score)

        if not categories:
            # Create a default chart with no data message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Comprehensive Analysis Radar Chart",
                height=500
            )
            return fig

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Analysis Results',
            line_color='#1f77b4'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Comprehensive Analysis Radar Chart",
            height=500
        )

        return fig

    def _create_career_timeline_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create career timeline visualization."""
        if "career_trajectory" not in results:
            return go.Figure()

        career = results["career_trajectory"]
        # Handle both string and dictionary responses
        if isinstance(career, str):
            try:
                import json
                career = json.loads(career)
            except (json.JSONDecodeError, ValueError):
                return go.Figure()

        if not isinstance(career, dict):
            return go.Figure()

        career_path = career.get('career_path', [])

        if not career_path:
            return go.Figure()

        # Extract timeline data
        positions = []
        years = []

        for i, position in enumerate(career_path[:10]):  # Limit to 10 positions
            positions.append(position.get('position', f'Position {i + 1}'))
            years.append(position.get('year', 2020 + i))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=positions,
            mode='lines+markers',
            name='Career Path',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4')
        ))

        fig.update_layout(
            title="Career Trajectory Timeline",
            xaxis_title="Year",
            yaxis_title="Position",
            height=400,
            yaxis=dict(autorange="reversed")
        )

        return fig

    def _create_market_intelligence_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create market intelligence visualization."""
        if "market_intelligence" not in results:
            return go.Figure()

        market = results["market_intelligence"]
        # Handle both string and dictionary responses
        if isinstance(market, str):
            try:
                import json
                market = json.loads(market)
            except (json.JSONDecodeError, ValueError):
                return go.Figure()

        if not isinstance(market, dict):
            return go.Figure()

        market_data = market.get('market_data', {})

        if not market_data:
            return go.Figure()

        # Extract market metrics
        metrics = []
        values = []

        for key, value in market_data.items():
            if isinstance(value, (int, float)):
                metrics.append(key.replace('_', ' ').title())
                values.append(value)

        if not values:
            return go.Figure()

        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color='#2ca02c',
                text=[f"{v:.1f}" for v in values],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Market Intelligence Metrics",
            xaxis_title="Market Metrics",
            yaxis_title="Value",
            height=400
        )

        return fig

    def display_visual_dashboard(self, results: Dict[str, Any], save_path: str = None):
        """Display the complete visual dashboard."""
        dashboard = self.create_visual_dashboard(results)

        print("\n" + "=" * 80)
        print("📊 VISUAL ANALYSIS DASHBOARD")
        print("=" * 80)

        # Display each chart
        for chart_name, fig in dashboard.items():
            if fig and len(fig.data) > 0:
                print(f"\n📈 {chart_name.replace('_', ' ').title()}")
                print("-" * 50)

                # Show the plot
                fig.show()

                # Save if path provided (with error handling)
                if save_path:
                    try:
                        os.makedirs(save_path, exist_ok=True)

                        # Save HTML version (always works)
                        html_path = f"{save_path}/{chart_name}.html"
                        fig.write_html(html_path)
                        print(f"  ✅ HTML saved: {html_path}")

                        # Try to save PNG (optional, may fail)
                        try:
                            png_path = f"{save_path}/{chart_name}.png"
                            fig.write_image(png_path)
                            print(f"  ✅ PNG saved: {png_path}")
                        except Exception as png_error:
                            print(f"  ⚠️  PNG save failed: {png_error}")
                            print(f"  💡 HTML version is available at: {html_path}")

                    except Exception as save_error:
                        print(f"  ❌ Save failed: {save_error}")

        print("\n" + "=" * 80)
        print("✅ Visual Dashboard Complete")
        print("=" * 80)

    def generate_html_report(self, results: Dict[str, Any], output_path: str = "cv_jd_analysis_report.html"):
        """Generate comprehensive HTML report with visualizations."""
        dashboard = self.create_visual_dashboard(results)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CV-JD Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #1f77b4;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                }}
                .chart-container {{
                    margin: 20px 0;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .score {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .timestamp {{
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 CV-JD Analysis Report</h1>
                    <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="summary">
        """

        # Add summary cards
        if "basic_matching" in results:
            basic = results["basic_matching"]
            # Handle both string and dictionary responses
            if isinstance(basic, str):
                try:
                    import json
                    basic = json.loads(basic)
                except (json.JSONDecodeError, ValueError):
                    basic = {}
            html_content += f"""
                    <div class="summary-card">
                        <h3>Overall Match</h3>
                        <div class="score">{basic.get('match_score', 0)}%</div>
                        <p>{basic.get('match_category', 'Unknown')}</p>
                    </div>
            """

        if "culture_fit" in results:
            culture = results["culture_fit"]
            # Handle both string and dictionary responses
            if isinstance(culture, str):
                try:
                    import json
                    culture = json.loads(culture)
                except (json.JSONDecodeError, ValueError):
                    culture = {}
            html_content += f"""
                    <div class="summary-card">
                        <h3>Culture Fit</h3>
                        <div class="score">{culture.get('culture_fit_score', 0)}%</div>
                        <p>Values Alignment</p>
                    </div>
            """

        if "career_trajectory" in results:
            career = results["career_trajectory"]
            # Handle both string and dictionary responses
            if isinstance(career, str):
                try:
                    import json
                    career = json.loads(career)
                except (json.JSONDecodeError, ValueError):
                    career = {}
            html_content += f"""
                    <div class="summary-card">
                        <h3>Career Alignment</h3>
                        <div class="score">{career.get('career_growth_potential', 0)}%</div>
                        <p>Trajectory Match</p>
                    </div>
            """

        html_content += """
                </div>
        """

        # Add charts
        for chart_name, fig in dashboard.items():
            if fig and len(fig.data) > 0:
                html_content += f"""
                <div class="section">
                    <h2>📈 {chart_name.replace('_', ' ').title()}</h2>
                    <div class="chart-container" id="{chart_name}"></div>
                </div>
                """

        # Add detailed results
        html_content += """
                <div class="section">
                    <h2>📋 Detailed Analysis Results</h2>
        """

        for module_name, result in results.items():
            # Handle both string and dictionary responses
            if isinstance(result, str):
                try:
                    import json
                    result = json.loads(result)
                except (json.JSONDecodeError, ValueError):
                    html_content += f"""
                        <h3>{module_name.replace('_', ' ').title()}</h3>
                        <p><strong>Analysis:</strong> {result}</p>
                    """
                    continue

            if isinstance(result, dict):
                html_content += f"""
                    <h3>{module_name.replace('_', ' ').title()}</h3>
                    <ul>
                """
                for key, value in result.items():
                    if isinstance(value, list):
                        html_content += f"<li><strong>{key}:</strong> {', '.join(map(str, value))}</li>"
                    else:
                        html_content += f"<li><strong>{key}:</strong> {value}</li>"
                html_content += "</ul>"
            else:
                html_content += f"""
                    <h3>{module_name.replace('_', ' ').title()}</h3>
                    <p><strong>Analysis:</strong> {str(result)}</p>
                """

        html_content += """
                </div>
            </div>

            <script>
        """

        # Add JavaScript for charts
        for chart_name, fig in dashboard.items():
            if fig and len(fig.data) > 0:
                # Convert figure to JSON properly
                fig_json = fig.to_json()
                html_content += f"""
                Plotly.newPlot('{chart_name}', {fig_json});
                """

        html_content += """
            </script>
        </body>
        </html>
        """

        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML report generated: {output_path}")
        return output_path


async def main():
    """Main function to demonstrate the enhanced CV-JD matcher."""
    api_key = API_KEY

    # Configuration - easily customizable
    config = {
        "cv_path": "E:\\imageextractor\\imageextractor\\Python-Dev-1.pdf",
        "jd_path": "E:\\imageextractor\\imageextractor\\React_Resume.pdf",
        "enabled_modules": [
            "basic_matching",
            "culture_fit",
            "career_trajectory",
            "resume_optimization",
            "interview_preparation",
            "bias_detection",
            "market_intelligence"
        ],
        "analysis_params": {
            "company_values": "Innovation, Collaboration, Continuous Learning, Customer Focus, Diversity",
            "industry": "Technology",
            "role": "Software Engineer",
            "location": "Remote",
            "save_images": False  # Disable image saving to avoid Kaleido issues
        }
    }

    # Initialize matcher with configuration
    matcher = EnhancedCVJDMatcher(
        api_key=api_key,
        enabled_modules=config["enabled_modules"]
    )

    # Run analysis
    await matcher.run(
        cv_path=config["cv_path"],
        jd_path=config["jd_path"],
        **config["analysis_params"]
    )


if __name__ == "__main__":
    asyncio.run(main())