import asyncio
import json
import os
import re
import logging
import traceback
import cv2
from typing import Dict, Any, List, Optional

import fitz
import numpy as np
import pytesseract
from PIL import Image
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict


class FieldMapping(BaseModel):
    """Represents a mapping between OCR text, JSON field, and PDF field"""
    model_config = ConfigDict(extra='allow')  # Allow extra fields

    ocr_text: Optional[str] = Field(default=None, description="Text extracted from OCR")
    json_field: Optional[str] = Field(default=None, description="Corresponding JSON field")
    pdf_field: Optional[str] = Field(default=None, description="Corresponding PDF field UUID")
    confidence: float = Field(default=0.0, description="Match confidence")
    suggested_value: Any = Field(default=None, description="Value to be filled")


class AdvancedPDFFiller:
    def __init__(self, api_key: str):
        """
        Initialize the PDF filler with enhanced logging and AI agent

        :param api_key: API key for the AI model
        """
        # Configure logging with more detailed formatting
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_filler_debug.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize AI agent with more specific instructions
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt="""
            You are an expert at extracting and mapping information from PDFs to JSON data.

            Matching Guidelines:
            1. Analyze OCR text carefully
            2. Match to most relevant JSON field
            3. Provide clear reasoning for matches
            4. Return structured JSON output

            Output Format (CRITICAL - MUST BE VALID JSON):
            {
                "matches": [
                    {
                        "ocr_text": "...",
                        "json_field": "...",
                        "pdf_field": "...",
                        "confidence": 0.85,
                        "suggested_value": "..."
                    }
                ]
            }

            IMPORTANT: Ensure the returned JSON is ALWAYS valid and matches this structure.
            """
        )

    async def match_ocr_to_json(self, ocr_results: List[Dict[str, Any]], json_data: Dict[str, Any]) -> List[
        FieldMapping]:
        """
        Match OCR text to JSON fields using AI with robust error handling

        :param ocr_results: List of OCR extracted text elements
        :param json_data: Input JSON data
        :return: List of field mappings
        """
        self.logger.info("Starting OCR to JSON matching process")

        try:
            # Flatten JSON for easier matching
            flat_json = self._flatten_json(json_data)

            # Prepare detailed matching prompt
            prompt = f"""
            Perform advanced text matching with the following constraints:

            OCR Text Elements:
            {json.dumps(ocr_results, indent=2)}

            Available JSON Fields:
            {json.dumps(flat_json, indent=2)}

            Task:
            1. Carefully match OCR text to most appropriate JSON field
            2. Calculate match confidence (0-1 scale)
            3. Suggest appropriate field values
            4. Return matches in EXACTLY this JSON format:

            {{
                "matches": [
                    {{
                        "ocr_text": "Example Text",
                        "json_field": "matching.field.name",
                        "pdf_field": "optional-uuid",
                        "confidence": 0.85,
                        "suggested_value": "Extracted Value"
                    }}
                ]
            }}

            CRITICAL: Ensure valid JSON output with "matches" array.
            """

            # Log the prompt for debugging
            self.logger.debug(f"Matching Prompt: {prompt}")

            # Get AI matching results with explicit error handling
            try:
                response = await self.agent.run(prompt)
                self.logger.debug(f"Raw AI Response: {response.data}")

                # Extensive parsing attempts
                parsed_matches = self._parse_ai_response(response.data)

                return parsed_matches

            except Exception as parse_error:
                self.logger.error(f"Critical parsing error: {parse_error}")
                self.logger.error(f"Problematic response: {response.data}")

                # Fallback parsing attempts
                try:
                    # Try direct JSON parsing with error tracking
                    raw_data = json.loads(response.data)
                    matches = raw_data.get('matches', [])

                    parsed_matches = [
                        FieldMapping(
                            ocr_text=match.get('ocr_text', ''),
                            json_field=match.get('json_field', ''),
                            pdf_field=match.get('pdf_field', ''),
                            confidence=match.get('confidence', 0.0),
                            suggested_value=match.get('suggested_value', None)
                        ) for match in matches
                    ]

                    return parsed_matches

                except json.JSONDecodeError as json_error:
                    self.logger.critical(f"Unrecoverable JSON parsing error: {json_error}")
                    # Attempt to extract JSON-like content
                    matches = self._extract_json_like_content(response.data)
                    return matches

        except Exception as e:
            self.logger.error(f"Unexpected error in field matching: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    def _parse_ai_response(self, response_data: str) -> List[FieldMapping]:
        """
        Robust parsing of AI response with multiple fallback strategies

        :param response_data: Raw AI response string
        :return: List of parsed field mappings
        """
        # Attempt to extract JSON from various formats
        json_patterns = [
            r'```json\s*([\s\S]*?)```',  # Code block with json
            r'```\s*([\s\S]*?)```',  # Generic code block
            r'\{.*\}',  # JSON-like content
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_data, re.DOTALL)
            if match:
                try:
                    # Try parsing the extracted content
                    data = json.loads(match.group(1).strip())

                    # Validate matches structure
                    matches = data.get('matches', [])

                    # Convert to FieldMapping with robust validation
                    parsed_matches = []
                    for match in matches:
                        try:
                            parsed_match = FieldMapping(**match)
                            parsed_matches.append(parsed_match)
                        except ValidationError as val_err:
                            self.logger.warning(f"Validation error for match: {val_err}")

                    return parsed_matches

                except json.JSONDecodeError as json_err:
                    self.logger.warning(f"JSON parsing failed: {json_err}")

        # If all parsing attempts fail
        self.logger.error("Could not parse AI response")
        return []

    def _extract_json_like_content(self, text: str) -> List[FieldMapping]:
        """
        Extract JSON-like content as a last resort parsing method

        :param text: Input text potentially containing JSON-like content
        :return: List of parsed field mappings
        """
        try:
            # Use regex to find potential JSON-like structures
            json_pattern = r'\{[^{}]*"ocr_text":[^{}]*"json_field":[^{}]*\}'
            matches = re.findall(json_pattern, text)

            parsed_matches = []
            for match_text in matches:
                try:
                    match_data = json.loads(match_text)
                    parsed_match = FieldMapping(**match_data)
                    parsed_matches.append(parsed_match)
                except Exception as e:
                    self.logger.warning(f"Could not parse individual match: {e}")

            return parsed_matches

        except Exception as e:
            self.logger.error(f"Fallback JSON extraction failed: {e}")
            return []

    # ... [rest of the previous implementation remains the same]


async def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Example usage with detailed error handling
    try:
        API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')
        pdf_path = "input.pdf"
        json_path = "form_data.json"
        output_pdf = "filled_output.pdf"

        # Initialize filler
        filler = AdvancedPDFFiller(API_KEY)

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Extract OCR text
        ocr_results = await filler.extract_ocr_text(pdf_path)

        # Match OCR to JSON with comprehensive logging
        field_matches = await filler.match_ocr_to_json(ocr_results, json_data)

        # Log matching results
        logging.info(f"Found {len(field_matches)} field matches")
        for match in field_matches:
            logging.info(f"Match: {match}")

        # Fill PDF
        success = filler.fill_pdf_fields(pdf_path, field_matches, output_pdf)

        print("PDF Filling", "Successful" if success else "Failed")

    except Exception as e:
        logging.error(f"Critical error in PDF processing: {e}")
        logging.debug(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())