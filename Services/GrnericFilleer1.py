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
    async def extract_pdf_fields_with_labels(self, pdf_path: str) -> Dict[str, Dict]:
        """Extracts fillable fields with their associated labels."""
        print("🔍 Extracting fields with labels...")
        doc = fitz.open(pdf_path)
        fields = {}

        # First pass - get all widgets
        all_widgets = {}
        for page_num, page in enumerate(doc, start=0):
            for widget in page.widgets():
                if widget.field_name:
                    field_name = widget.field_name.strip()
                    # Store widget info including rect coordinates
                    all_widgets[field_name] = {
                        "page": page_num,
                        "rect": widget.rect,
                        "field_type": widget.field_type,
                        "field_value": widget.field_value
                    }

        # Second pass - find labels near widgets
        for field_name, widget_info in all_widgets.items():
            page = doc[widget_info["page"]]
            rect = widget_info["rect"]

            # Look for text before the field (typically labels are to the left or above)
            label_rect = fitz.Rect(rect.x0 - 200, rect.y0 - 50, rect.x0, rect.y0 + rect.height)
            label_text = page.get_text("text", clip=label_rect).strip()

            # If nothing found, try above with wider area
            if not label_text:
                label_rect = fitz.Rect(rect.x0 - 300, rect.y0 - 100, rect.x0 + rect.width, rect.y0)
                label_text = page.get_text("text", clip=label_rect).strip()

            # Clean up the label text
            if label_text:
                # Remove common endings like colons, etc.
                label_text = re.sub(r'[:_*\[\]\(\)]+\s*$', '', label_text).strip()

            widget_info["label"] = label_text
            fields[field_name] = widget_info

        doc.close()
        return fields

    def fill_pdf_immediately(self, template_pdf: str, output_pdf: str, matches: List[FieldMatch],
                             pdf_fields: Dict[str, Dict]):
        """Fills PDF form fields using PyMuPDF (fitz) for better compatibility."""
        doc = fitz.open(template_pdf)
        filled_fields = {}

        for match in matches:
            if match.pdf_field and match.suggested_value is not None:
                # Check if the field exists in our extracted fields
                if match.pdf_field in pdf_fields:
                    field_info = pdf_fields[match.pdf_field]
                    page_num = field_info["page"]
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
                else:
                    print(f"⚠️ Field not found in PDF: {match.pdf_field}")

        doc.save(output_pdf)
        doc.close()

        print(f"✅ Successfully filled {len(filled_fields)} fields: {list(filled_fields.keys())[:5]}...")

        self.verify_pdf_filled(output_pdf)
        return filled_fields

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3):
        """Matches fields using AI and fills them immediately across multiple pages."""
        pdf_fields = await self.extract_pdf_fields_with_labels(pdf_path)
        flat_json = self.flatten_json(json_data)

        # Create a simplified version of the fields dictionary for the prompt
        pdf_field_names = list(pdf_fields.keys())

        prompt = PDF_FIELD_MATCHING_PROMPT2.format(
            json_data=json.dumps(flat_json, indent=2),
            pdf_fields=json.dumps(pdf_field_names, indent=2)
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
            return False

        filled_fields = self.fill_pdf_immediately(pdf_path, output_pdf, matches, pdf_fields)
        return bool(filled_fields)
    # async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
    #                                 max_retries: int = 3):
    #     """Matches fields using AI and fills them immediately across multiple pages."""
    #     pdf_fields = await self.extract_pdf_fields_with_labels(pdf_path)
    #     flat_json = self.flatten_json(json_data)
    #
    #     prompt = PDF_FIELD_MATCHING_PROMPT2.format(
    #         json_data=json.dumps(flat_json, indent=2),
    #         pdf_fields=json.dumps(list(pdf_fields.keys()), indent=2)
    #     )
    #
    #     matches = []
    #     for attempt in range(max_retries):
    #         response = await self.agent.run(prompt)
    #         matches = self.parse_ai_response(response.data)
    #
    #         if matches:
    #             break
    #
    #         print(f"Attempt {attempt + 1}/{max_retries} failed to get valid matches. Retrying...")
    #
    #     if not matches:
    #         print("⚠️ No valid field matches were found after all attempts.")
    #
    #     self.fill_pdf_immediately(pdf_path, output_pdf, matches, pdf_fields)

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

    # def fill_pdf_immediately(self, template_pdf: str, output_pdf: str, matches: List[FieldMatch],
    #                          pdf_fields: Dict[str, int]):
    #     """Fills PDF form fields using PyMuPDF (fitz) for better compatibility."""
    #
    #     doc = fitz.open(template_pdf)
    #
    #
    #     filled_fields = {}
    #
    #     for match in matches:
    #         if match.pdf_field and match.suggested_value is not None:
    #             page_num = pdf_fields.get(match.pdf_field, 0)
    #             page = doc[page_num]
    #
    #             for widget in page.widgets():
    #                 if widget.field_name == match.pdf_field:
    #                     print(f"✍️ Filling: '{match.suggested_value}' → '{match.pdf_field}' (Page {page_num + 1})")
    #                     try:
    #                         # Update the field value
    #                         widget.field_value = str(match.suggested_value)
    #                         # Apply the changes to the widget
    #                         widget.update()
    #                         filled_fields[match.pdf_field] = match.suggested_value
    #                     except Exception as e:
    #                         print(f"⚠️ Error filling {match.pdf_field}: {e}")
    #                     break
    #
    #
    #     doc.save(output_pdf)
    #     doc.close()
    #
    #     print(f"✅ Successfully filled {len(filled_fields)} fields: {list(filled_fields.keys())[:5]}...")
    #
    #     self.verify_pdf_filled(output_pdf)
    #     return filled_fields

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
    output_pdf = "D:\\demo\\Services\\fill_smart15.pdf"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

    if success:
        print(f"✅ PDF successfully processed: {output_pdf}")
    else:
        print(f"❌ PDF processing failed. Please check the output file and logs.")


if __name__ == "__main__":
    asyncio.run(main())