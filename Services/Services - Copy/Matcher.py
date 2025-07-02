from typing import Dict, Any, List, Optional, Set
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from Common.constants import *

class AnalysisResult(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    matching_skills: List[str]
    missing_requirements: List[str]
    match_category: str
    overall_assessment: str
    key_strengths: List[str]
    improvement_suggestions: List[str]
    technical_analysis: Optional[Dict[str, Any]]
    experience_match: Optional[Dict[str, Any]]

class ExtractorService:
    def __init__(self, agent, max_retries: int = 3):
        self.agent = agent
        self.max_retries = max_retries

    async def extract_with_retry(self, cv_text: str, jd_text: str, extraction_type: str) -> Dict[str, Any]:
        extraction_prompts = {
            "skills": """
            Analyze the CV and job requirements in detail.
            Provide a JSON response with:
            {
                "matching_skills": [list of skills present in both CV and JD],
                "missing_skills": [list of required skills not found in CV],
                "key_strengths": [list of notable skills from CV],
                "skill_match_score": numeric score between 0-100 based on skill alignment,
                "skill_assessment": detailed explanation of skill matching
            }
            Consider both explicit and implicit skills. Evaluate skill proficiency where possible.
            """,
            "experience": """
            Analyze experience requirements and qualifications deeply.
            Provide a JSON response with:
            {
                "years_required": exact number (numeric),
                "years_matched": exact number (numeric),
                "experience_score": numeric score between 0-100 based on experience match,
                "experience_assessment": detailed analysis of experience alignment,
                "key_experiences": [list of relevant experiences],
                "experience_gaps": [list of areas needing more experience]
            }
            Consider quality and relevance of experience, not just years.
            """,
            "technical": """
            Perform comprehensive technical analysis and provide a JSON response with:
            {
                "technical_match": {
                    "matching_skills": [list],
                    "missing_skills": [list],
                    "proficiency_levels": object,
                    "technical_score": numeric score between 0-100
                },
                "technical_assessment": detailed evaluation of technical capabilities,
                "improvement_areas": [list of technical areas to improve],
                "technical_strengths": [list of standout technical capabilities]
            }
            Evaluate depth of technical knowledge and practical application.
            """
        }

        for attempt in range(self.max_retries):
            try:
                prompt = f"{extraction_prompts[extraction_type]}\n\nCV:\n{cv_text}\n\nJD:\n{jd_text}"
                response: RunResult = await self.agent.run(prompt)
                return self._parse_json_response(response.data)
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed for {extraction_type}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._get_default_result(extraction_type)
            await asyncio.sleep(1)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text)

    def _get_default_result(self, extraction_type: str) -> Dict[str, Any]:
        defaults = {
            "skills": {
                "matching_skills": [],
                "missing_skills": [],
                "key_strengths": [],
                "skill_match_score": 0,
                "skill_assessment": "Failed to extract skills data",
                "extraction_error": True
            },
            "experience": {
                "years_required": 0,
                "years_matched": 0,
                "experience_score": 0,
                "experience_assessment": "Failed to extract experience data",
                "key_experiences": [],
                "experience_gaps": [],
                "extraction_error": True
            },
            "technical": {
                "technical_match": {
                    "matching_skills": [],
                    "missing_skills": [],
                    "proficiency_levels": {},
                    "technical_score": 0
                },
                "technical_assessment": "Failed to extract technical data",
                "improvement_areas": [],
                "technical_strengths": [],
                "extraction_error": True
            }
        }
        return defaults[extraction_type]

class AutonomousCVJDMatcher:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", max_retries: int = 3):
        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = """You are an expert CV analyzer with deep knowledge of industry requirements,
        technical skills, and career progression. Provide detailed, nuanced analysis of candidate fit,
        considering both explicit and implicit qualifications. Focus on practical relevance and potential
        for success in the role."""

        self.cv_jd_agent = Agent(
            model=self.llm,
            system_prompt=self.system_prompt,
            retries=3,
        )
        self.extractor_service = ExtractorService(self.cv_jd_agent, max_retries)

    async def analyze_overall_match(self, cv_text: str, jd_text: str,
                                  skills_data: Dict, experience_data: Dict,
                                  technical_data: Dict) -> Dict[str, Any]:
        prompt = f"""
        Analyze the overall match between the candidate and job requirements.
        Consider the following data points:

        Skills Match:
        {json.dumps(skills_data, indent=2)}

        Experience Match:
        {json.dumps(experience_data, indent=2)}

        Technical Match:
        {json.dumps(technical_data, indent=2)}

        Provide a JSON response with:
        {{
            "match_score": overall match score (0-100),
            "match_category": one of ["Best Match", "Good Match", "Average Match", "Poor Match"],
            "overall_assessment": detailed evaluation of candidate fit,
            "key_strengths": [main advantages of this candidate],
            "improvement_suggestions": [specific areas for development],
            "hiring_recommendation": specific recommendation with rationale
        }}

        Base your analysis on both quantitative and qualitative factors.
        Consider cultural fit and potential for growth.
        """

        response: RunResult = await self.cv_jd_agent.run(prompt)
        return self._parse_json_response(response.data)

    async def run(self, cv_path: str, jd_path: str) -> Optional[Dict[str, Any]]:
        try:
            print("\n📄 Processing documents...")

            cv_text = await self._extract_text(cv_path)
            jd_text = await self._extract_text(jd_path)

            if not cv_text or not jd_text:
                return None

            print("\n🔍 Starting Analysis...")

            skills_task = self.extractor_service.extract_with_retry(cv_text, jd_text, "skills")
            experience_task = self.extractor_service.extract_with_retry(cv_text, jd_text, "experience")
            technical_task = self.extractor_service.extract_with_retry(cv_text, jd_text, "technical")

            skills_data, experience_data, technical_data = await asyncio.gather(
                skills_task, experience_task, technical_task
            )

            overall_analysis = await self.analyze_overall_match(
                cv_text, jd_text, skills_data, experience_data, technical_data
            )

            result = {
                **overall_analysis,
                "matching_skills": skills_data["matching_skills"],
                "missing_requirements": skills_data["missing_skills"],
                "experience_match": experience_data,
                "technical_analysis": technical_data,
                "timestamp": datetime.now().isoformat()
            }

            self._print_analysis_results(result)
            return result

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return None

    async def _extract_text(self, file_path: str) -> Optional[str]:
        try:
            from Factory.DocumentFactory import ExtractorFactory
            extractor = ExtractorFactory().get_extractor(file_path)
            text = await extractor.extract_text(file_path)
            print(f"✓ Successfully extracted content from {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return None

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text)

    def _print_analysis_results(self, result: Dict[str, Any]):
        print("\n📊 Analysis Results:")
        print(f"🎯 Match Score: {result['match_score']}%")
        print(f"📝 Category: {result['match_category']}")
        print(f"\n📋 Overall Assessment:\n{result['overall_assessment']}")

        print("\n💪 Key Strengths:")
        for strength in result['key_strengths']:
            print(f"• {strength}")

        print("\n Improvement Suggestions:")
        for suggestion in result['improvement_suggestions']:
            print(f"• {suggestion}")

        if 'hiring_recommendation' in result:
            print(f"\nRecommendation:\n{result['hiring_recommendation']}")

async def main():
    api_key = API_KEY
    cv_path = "D:\\demo\\Python-Dev-1.pdf"
    jd_path = "D:\\demo\\Python-Lead-Dev.pdf"

    matcher = AutonomousCVJDMatcher(api_key)
    await matcher.run(cv_path, jd_path)

if __name__ == "__main__":
    asyncio.run(main())