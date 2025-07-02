import fitz
import asyncio
import os
import json
from typing import Dict, Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from pydantic_ai.tools import ToolDefinition

from Common.constants import API_KEY


async def extract_pdf_text_tool(ctx: RunContext, file_path: str) -> str:
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


async def fetch_website_content_tool(ctx: RunContext, url: str) -> str:
    """Fetches content from a URL using an external tool."""

    return f"Fetched content from {url}"


class CVJDMatcher:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initializes the JD-CV Matcher using OpenAI API with pydantic_ai.
        """
        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = """
        You are an expert recruiter AI analyzing resumes and job descriptions.
        Your task is to:
        - Extract and match the relevant skills, qualifications, and experiences from the CV and JD.
        - Exclude any irrelevant information such as POS tags, common words, and other non-skill-related content.
        - Identify missing skills or gaps in the CV based on JD requirements.
        - Provide a match score (0-100%) and a comprehensive assessment.
        """

        self.cv_jd_agent = Agent(
            model=self.llm,
            system_prompt=self.system_prompt,
            retries=2,
        )

    async def analyze_cv_and_jd(self, cv_text: str, jd_text: str) -> Dict[str, Any]:

        prompt = f"""
        Compare the following CV and JD and provide the following details:
        - Match Score (percentage) as match_score
        - Matching Skills (relevant skills only) as matching_skills
        - Missing Requirements (relevant skills only) as missing_requirements
        - Match Category (Best Match / Good Match / Average Match / Poor Match) as match_category
        - Overall Assessment as overall_assessment

        Exclude any irrelevant information such as POS tags, common words, and other non-skill-related content and return all of this in json format.

        CV Text:
        {cv_text}

        JD Text:
        {jd_text}
        """
        try:
            response: RunResult = await self.cv_jd_agent.run(prompt)
            return response.data
        except Exception as e:
            print(f"Error in AI-based analysis: {str(e)}")
            return {"error": str(e)}

    def _normalize_keys(self, response: Dict[str, Any]) -> Dict[str, Any]:

        normalized_response = {}
        for key, value in response.items():
            normalized_key = key.lower().replace(" ", "_")
            normalized_response[normalized_key] = value
        return normalized_response

    def _parse_response(self, response: str) -> Dict[str, Any]:

        try:
            response_data = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
            normalized_data = self._normalize_keys(response_data)
            match_score = normalized_data.get("match_score", 0)
            if isinstance(match_score, str) and match_score.isdigit():
                match_score = float(match_score)
            matching_skills = normalized_data.get("matching_skills", []) or normalized_data.get("matchingskills", [])

            return {
                "match score": match_score,
                "match category": normalized_data.get("match_category"),
                "matching skills": matching_skills,
                "missing requirements": normalized_data.get("missing_requirements", []),
                "overall assessment": normalized_data.get("overall_assessment", "")
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            return {"error": str(e)}

    async def process_and_match(self, cv_path: str, jd_path: str, runs: int = 1) -> Dict[str, Any]:
        """
        Processes the CV and JD PDF files and matches them using the agent.
        """
        cv_text = await extract_pdf_text_tool(None, cv_path)
        jd_text = await extract_pdf_text_tool(None, jd_path)

        if not cv_text.strip() or not jd_text.strip():
            return {"error": "No text extracted from one or both PDF files."}

        best_match_score = 0
        best_match_result = None

        for _ in range(runs):
            analysis_result = await self.analyze_cv_and_jd(cv_text, jd_text)
            if "error" in analysis_result:
                return analysis_result
            parsed_result = self._parse_response(analysis_result)
            match_score = parsed_result.get("match score", 0)

            print(f"Match Score from this run: {match_score}")

            if match_score:
                try:
                    match_score = float(match_score)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_result = parsed_result
                except ValueError:
                    print(f"Warning: Invalid match score received: {match_score}")

        if best_match_result:
            best_match_result["match score"] = f"{best_match_score:.2f}%"
        else:
            best_match_result = {"match score": "0.00%"}

        return best_match_result

    async def run(self, cv_path: str, jd_path: str):
        """
        Main function to run the CV and JD matching process.
        """
        if not (os.path.exists(cv_path) and os.path.exists(jd_path)):
            print("Error: One or both provided PDF paths do not exist.")
            return

        print("\n=== Starting CV and JD Analysis ===")

        result = await self.process_and_match(cv_path, jd_path)

        if "error" in result:
            print(result["error"])
        else:
            analysis_result = result

            print(f"Best Match Result: {analysis_result}")

            if analysis_result is not None:
                match_score = analysis_result.get("match score", "0")
                print(f"Best Match Score: {match_score}%")

                print("\nMatching Skills:")
                for skill in analysis_result.get("matching skills", []):
                    print(f"- {skill}")

                print("\nMissing Requirements:")
                for req in analysis_result.get("missing requirements", []):
                    print(f"- {req}")

                print("\nOverall Assessment:")
                print(analysis_result.get("overall assessment", ""))

                print("\n=== Analysis Complete ===")
            else:
                print("Analysis Result is None")


async def main():
    api_key = API_KEY
    jd_path = "D:\\demo\\React_JS_5_Years.pdf"
    cv_path = "D:\\demo\\React_Developer_5_Years_Resume.pdf"

    matcher = CVJDMatcher(api_key)
    await matcher.run(cv_path, jd_path)


if __name__ == "__main__":
    asyncio.run(main())
