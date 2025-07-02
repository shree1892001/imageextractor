from typing import Dict, Any, List, Optional
import asyncio
import json
from collections import deque
from datetime import datetime
from functools import wraps

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult

from Factory.DocumentFactory import ExtractorFactory
from Common.constants import *

class MatchResult(BaseModel):
    match_score: int
    matching_skills: List[str]
    missing_requirements: List[str]
    match_category: str
    overall_assessment: str

def cv_analysis_tool(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        print("\n🔧 Executing CV Analysis Tool...")

        tool_params = {
            "name": "cv_analysis_tool",
            "description": "Analyzes CV and JD documents to provide detailed matching analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_text": {"type": "string", "description": "The text content of the CV"},
                    "jd_text": {"type": "string", "description": "The text content of the job description"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["basic", "detailed", "technical"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["cv_text", "jd_text"]
            }
        }

        try:

            execution_start = datetime.now()
            print(f"📋 Tool Parameters: {tool_params}")

            result = await func(self, *args, **kwargs)

            execution_time = (datetime.now() - execution_start).total_seconds()
            print(f"⏱️ Tool Execution Time: {execution_time:.2f} seconds")

            if isinstance(result, dict):
                result["tool_metadata"] = {
                    "tool_name": tool_params["name"],
                    "execution_time": execution_time,
                    "timestamp": execution_start.isoformat()
                }

            return result

        except Exception as e:
            print(f"❌ Tool Execution Error: {str(e)}")
            return {"error": str(e), "tool_name": tool_params["name"]}

    return wrapper

class AutonomousCVJDMatcher:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = SYSTEM_PROMPT_MATCHER
        self.cv_jd_agent = Agent(
            model=self.llm,
            system_prompt=self.system_prompt,
            retries=3,
        )
        self.memory = []
        self.task_queue = deque()
        self.analysis_history: List[Dict[str, Any]] = []

    @cv_analysis_tool
    async def _execute_tool(self, cv_text: str, jd_text: str, analysis_type: str = "detailed") -> Dict[str, Any]:

        prompt_templates = {
            "basic": """
                Compare the CV and JD and provide a basic match analysis in JSON format:
                {
                    "match_score": integer (0-100),
                    "match_category": "Best Match" | "Good Match" | "Average Match" | "Poor Match",
                    "overall_assessment": string
                }
            """,
            "detailed": """
                Provide a comprehensive analysis of the CV and JD match in JSON format:
                {
                    "match_score": integer (0-100),
                    "matching_skills": list of strings,
                    "missing_requirements": list of strings,
                    "match_category": "Best Match" | "Good Match" | "Average Match" | "Poor Match",
                    "overall_assessment": string,
                    "improvement_suggestions": list of strings,
                    "key_strengths": list of strings
                }
            """,
            "technical": """
                Perform a technical skills-focused analysis in JSON format:
                {
                    "match_score": integer (0-100),
                    "technical_skills_match": {
                        "matching_skills": list of strings,
                        "missing_skills": list of strings,
                        "skill_proficiency_assessment": object
                    },
                    "experience_match": {
                        "years_required": number,
                        "years_matched": number,
                        "experience_assessment": string
                    },
                    "match_category": "Best Match" | "Good Match" | "Average Match" | "Poor Match",
                    "technical_gaps": list of strings
                }
            """
        }

        prompt = f"""
        {prompt_templates[analysis_type]}

        CV:
        {cv_text}

        JD:
        {jd_text}
        """

        try:
            response: RunResult = await self.cv_jd_agent.run(prompt)
            response_text = response.data.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text)

            result["analysis_type"] = analysis_type
            result["timestamp"] = datetime.now().isoformat()

            return result

        except json.JSONDecodeError:
            return {"error": "AI re   sponse is not valid JSON"}
        except Exception as e:
            return {"error": str(e)}

    @cv_analysis_tool
    async def analyze_cv_and_jd(self, cv_text: str, jd_text: str) -> Dict[str, Any]:


        detailed_result = await self._execute_tool(cv_text, jd_text, "detailed")

        if detailed_result.get("match_score", 0) < 70:
            technical_result = await self._execute_tool(cv_text, jd_text, "technical")

            merged_result = {
                **detailed_result,
                "technical_analysis": technical_result.get("technical_skills_match", {}),
                "experience_match": technical_result.get("experience_match", {}),
                "technical_gaps": technical_result.get("technical_gaps", [])
            }
            return merged_result

        return detailed_result

    async def run(self, cv_path: str, jd_path: str):
        try:

            cv_extractor = ExtractorFactory.get_extractor(cv_path)
            jd_extractor = ExtractorFactory.get_extractor(jd_path)

            print("\n🔍 Extracting text from files...")
            cv_text = await cv_extractor.extract_text(cv_path)
            jd_text = await jd_extractor.extract_text(jd_path)

            if not cv_text or not jd_text:
                print("❌ Error: No text extracted.")
                return

            print("\n🤖 Analyzing CV and JD...")
            match_result = await self.analyze_cv_and_jd(cv_text, jd_text)

            if "error" in match_result:
                print("❌ AI Error:", match_result["error"])
                return

            self.analysis_history.append(match_result)

            if "tool_metadata" in match_result:
                print("\n🔧 Tool Execution Details:")
                metadata = match_result["tool_metadata"]
                print(f"Tool: {metadata['tool_name']}")
                print(f"Execution Time: {metadata['execution_time']:.2f} seconds")
                print(f"Timestamp: {metadata['timestamp']}")

            print("\n📊 Analysis Results:")
            print(f"🎯 Match Score: {match_result.get('match_score')}%")
            print(f"📝 Category: {match_result.get('match_category')}")

            if "technical_analysis" in match_result:
                print("\n🔧 Technical Analysis:")
                print(f"✓ Matching Skills: {', '.join(match_result['technical_analysis'].get('matching_skills', []))}")
                print(f"✗ Missing Skills: {', '.join(match_result['technical_analysis'].get('missing_skills', []))}")

                if "experience_match" in match_result:
                    print(f"\n⏳ Experience Match:")
                    exp_match = match_result["experience_match"]
                    print(f"Required: {exp_match.get('years_required')} years")
                    print(f"Matched: {exp_match.get('years_matched')} years")

            print("\n💡 Improvement Suggestions:")
            for suggestion in match_result.get("improvement_suggestions", []):
                print(f"• {suggestion}")

            print("\n✅ Overall Assessment:")
            print(match_result.get("overall_assessment"))

            return match_result

        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e}")
        except ValueError as e:
            print(f"❌ Error: Invalid data - {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

async def main():
    api_key = API_KEY
    cv_path = "D:\\demo\\Python-Dev-1.pdf"
    jd_path = "D:\\demo\\Python-Lead-Dev.pdf"

    agent = AutonomousCVJDMatcher(api_key)
    await agent.run(cv_path, jd_path)

if __name__ == "__main__":
    asyncio.run(main())