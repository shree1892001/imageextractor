import asyncio
import json
import os
import re
from typing import Dict, Any, List

import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator


from Common.constants import *

API_KEYS = {
    "field_matcher": API_KEY_3,
}


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str
    confidence: float
    suggested_value: Any
    reasoning: str

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)


class MultiAgentFormFiller:
    def __init__(self):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["field_matcher"]),
            system_prompt="You are an expert at mapping PDF fields to JSON keys and filling them immediately."
        )

    # async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, int]:
    #     """Extracts all fillable fields from a multi-page PDF."""
    #     print("🔍 Extracting all fillable fields...")
    #     doc = fitz.open(pdf_path)
    #     fields = {}
    #
    #     for page_num, page in enumerate(doc, start=0):
    #         for widget in page.widgets():
    #             if widget.field_name:
    #                 field_name = widget.field_name.strip()
    #                 fields[field_name] = page_num  # Store the page number of the field
    #
    #     print(f"✅ Extracted {len(fields)} fields across {len(doc)} pages.")
    #     for field, page in fields.items():
    #         print(f" - Field: '{field}' (Page {page + 1})")
    #
    #     return fields
    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, int]:
        """Extracts all fillable fields from a multi-page PDF."""
        print("🔍 Extracting all fillable fields...")
        doc = fitz.open(pdf_path)
        fields = {}

        for page_num, page in enumerate(doc, start=0):
            for widget in page.widgets():
                if widget.field_name:
                    field_name = widget.field_name.strip()
                    sanitized_field_name = field_name # Remove special characters
                    fields[sanitized_field_name] = page_num

        if not fields:
            print("⚠️ No form fields found. Attempting text-based extraction...")
            for page_num, page in enumerate(doc):
                text = page.get_text("text")

                # Extract possible field names
                label_matches = re.findall(r"([\w\s,]+)[:]\s*", text)
                form_matches = re.findall(r"([\w\s,]+)(?:_+|\[\s*\]|\(\s*\))", text)
                all_matches = set(label_matches + form_matches)

                for match in all_matches:
                    field_name = match.strip()
                    if field_name and len(field_name) > 2:
                        sanitized_field_name =field_name
                        fields[sanitized_field_name] = page_num
                        print(f"✅ Extracted field: {sanitized_field_name} on page {page_num + 1}")

        print(f"✅ Extracted {len(fields)} fields across {len(doc)} pages.")
        doc.close()
        return fields
    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3):
        """Matches fields using AI and fills them immediately across multiple pages."""
        pdf_fields = await self.extract_pdf_fields(pdf_path)
        flat_json = self.flatten_json(json_data)

        prompt = PDF_FIELD_MATCHING_PROMPT2.format(
            json_data=json.dumps(flat_json, indent=2),
            pdf_fields=json.dumps(list(pdf_fields.keys()), indent=2)
        )

        matches = []
        for attempt in range(max_retries):
            response = await self.agent.run(prompt)
            matches = self.parse_ai_response(response.data)

            if matches:
                break

            print(f"Attempt {attempt + 1}/{max_retries} failed to get valid matches. Retrying...")

        if not matches:
            print("⚠️ No valid field matches were found after all attempts.")

        self.fill_pdf_immediately(pdf_path, output_pdf, matches, pdf_fields)

    def parse_ai_response(self, response_text: str) -> List[FieldMatch]:
        """Parses AI response and extracts valid JSON matches, handling missing fields."""
        response_text = re.sub(r"^```json", "", response_text).strip().rstrip("```")
        try:
            data = json.loads(response_text)
            matches = []

            for match in data.get("matches", []):
                match.setdefault("confidence", 1.0)
                match.setdefault("reasoning", "No reasoning provided.")

                try:
                    validated_match = FieldMatch(**match)
                    matches.append(validated_match)
                except Exception as e:
                    print(f"⚠️ Skipping malformed match: {match} | Error: {e}")

            return matches
        except json.JSONDecodeError:
            print("❌ AI returned invalid JSON. Retrying...")
            return []

    def fill_pdf_immediately(self, template_pdf: str, output_pdf: str, matches: List[FieldMatch],
                             pdf_fields: Dict[str, int]):
        """Fills PDF form fields using PyMuPDF (fitz) for better compatibility."""

        doc = fitz.open(template_pdf)


        filled_fields = {}

        for match in matches:
            if match.pdf_field and match.suggested_value is not None:
                page_num = pdf_fields.get(match.pdf_field, 0)
                page = doc[page_num]

                for widget in page.widgets():
                    if widget.field_name == match.pdf_field:
                        print(f"✍️ Filling: '{match.suggested_value}' → '{match.pdf_field}' (Page {page_num + 1})")
                        try:
                            # Update the field value
                            widget.field_value = str(match.suggested_value)
                            # Apply the changes to the widget
                            widget.update()
                            filled_fields[match.pdf_field] = match.suggested_value
                        except Exception as e:
                            print(f"⚠️ Error filling {match.pdf_field}: {e}")
                        break


        doc.save(output_pdf)
        doc.close()

        print(f"✅ Successfully filled {len(filled_fields)} fields: {list(filled_fields.keys())[:5]}...")

        self.verify_pdf_filled(output_pdf)
        return filled_fields

    def verify_pdf_filled(self, pdf_path: str) -> bool:
        """Verifies that the PDF has been filled correctly."""
        reader = PdfReader(pdf_path)
        fields = reader.get_fields()

        if not fields:
            print("⚠️ No fillable fields found. PDF may be flattened.")
            return False

        filled_fields = {k: v.get("/V") for k, v in fields.items() if v.get("/V")}
        print(f"✅ Filled {len(filled_fields)} fields: {list(filled_fields.keys())[:5]}...")
        return bool(filled_fields)

    def flatten_json(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flattens nested JSON objects into a flat dictionary."""
        items = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self.flatten_json(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(self.flatten_json(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = value
        return items



async def main():
    form_filler = MultiAgentFormFiller()
    template_pdf = "D:\\demo\\Services\\OhioLLC.pdf"
    json_path = "D:\\demo\\Services\\form_data1.json"
    output_pdf = "D:\\demo\\Services\\fill_smart14.pdf"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

    if success:
        print(f"✅ PDF successfully processed: {output_pdf}")
    else:
        print(f"❌ PDF processing failed. Please check the output file and logs.")


if __name__ == "__main__":
    asyncio.run(main())