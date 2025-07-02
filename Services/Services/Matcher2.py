from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
from functools import wraps
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from Common.constants import *
from Factory.DocumentFactory import ExtractorFactory


def cv_analysis_tool(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        print("\n\U0001F527 Executing CV Analysis Tool...")

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
            print(f"\U0001F4CB Tool Parameters: {tool_params}")

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
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", max_retries: int = 3):
        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = SYSTEM_PROMPT_MATCHER
        self.cv_jd_agent = Agent(model=self.llm, system_prompt=self.system_prompt, retries=max_retries)

    @cv_analysis_tool
    async def analyze_overall_match(self, cv_text: str, jd_text: str) -> Dict[str, Any]:
        prompt = f"""
        Perform a detailed CV and JD analysis.
        Provide a structured JSON output:
        {{
            "match_score": integer (0-100),
            "match_category": "Best Match" | "Good Match" | "Average Match" | "Poor Match",
            "overall_assessment": string,
            "key_strengths": list,
            "improvement_suggestions": list,
            "hiring_recommendation": string
        }}
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
            overall_analysis = await self.analyze_overall_match(cv_text, jd_text)

            if overall_analysis is None:
                print("❌ Analysis failed. No valid response.")
                return None

            result = {**overall_analysis, "timestamp": datetime.now().isoformat()}
            self._print_analysis_results(result)
            return result
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return None

    async def _extract_text(self, file_path: str) -> Optional[str]:
        try:
            extractor = ExtractorFactory().get_extractor(file_path)
            text = await extractor.extract_text(file_path)
            print(f"✓ Successfully extracted content from {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            if not response.strip():
                raise ValueError("Empty response received from API.")
            parsed_data = json.loads(response)

            required_keys = ["match_score", "match_category", "overall_assessment", "key_strengths",
                             "improvement_suggestions"]
            for key in required_keys:
                if key not in parsed_data:
                    raise KeyError(f"Missing key in API response: {key}")

            return parsed_data
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parsing Error: {str(e)}")
        except KeyError as e:
            print(f"❌ Missing Expected Key: {str(e)}")
        return None

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
