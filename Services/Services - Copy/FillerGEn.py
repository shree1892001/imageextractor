import asyncio
import json
import os
import re
import shutil
from typing import Dict, Any, List, Tuple, Optional

import fitz
import numpy as np
import cv2
import pytesseract
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator

from Common.constants import *

API_KEYS = {
    "field_matcher": API_KEY_1,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str  # or = None if you update the validator
    confidence: float
    suggested_value: Any
    reasoning: str

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)


class OCRFieldMatch(BaseModel):
    json_field: str
    ocr_text: str
    page_num: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    pdf_field: str  # or = None if you update the validator

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

        # No need to initialize PaddleOCR - pytesseract uses system installation

        self.matched_fields = {}

    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        """Extracts all fillable fields from a multi-page PDF with additional metadata."""
        print("🔍 Extracting all fillable fields...")
        doc = fitz.open(pdf_path)
        fields = {}

        for page_num, page in enumerate(doc, start=0):
            for widget in page.widgets():
                if widget.field_name:
                    field_name = widget.field_name.strip()
                    field_type = widget.field_type
                    field_rect = widget.rect
                    field_flags = widget.field_flags

                    fields[field_name] = {
                        "page_num": page_num,
                        "type": field_type,
                        "rect": [field_rect.x0, field_rect.y0, field_rect.x1, field_rect.y1],
                        "flags": field_flags,
                        "is_readonly": bool(field_flags & 1),
                        "current_value": widget.field_value
                    }

        print(f"✅ Extracted {len(fields)} fields across {len(doc)} pages.")
        for field, info in fields.items():
            readonly_status = "READ-ONLY" if info["is_readonly"] else "EDITABLE"
            print(f" - Field: '{field}' (Page {info['page_num'] + 1}) [{readonly_status}]")

        doc.close()
        return fields

    async def extract_ocr_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using pytesseract OCR with position information."""
        print("🔍 Extracting text using OCR...")
        doc = fitz.open(pdf_path)
        ocr_results = []

        for page_num in range(len(doc)):
            print(f"Processing OCR for page {page_num + 1}/{len(doc)}...")
            pix = doc[page_num].get_pixmap(alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Preprocess the image for better OCR results
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # First OCR attempt with adaptive threshold
            custom_config = r'--oem 3 --psm 11'
            data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

            # If no results or poor results, try with regular threshold
            if len(data['text']) == 0 or all(not text.strip() for text in data['text']):
                _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                data = pytesseract.image_to_data(threshold, config=custom_config, output_type=pytesseract.Output.DICT)

            # Process OCR results
            seen_texts = set()
            for i in range(len(data['text'])):
                text = data['text'][i].strip().lower()
                if not text or text in seen_texts or data['conf'][i] < 40:  # Filter by 40% confidence
                    continue

                seen_texts.add(text)
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                cleaned_text = ''.join(c for c in text if c.isprintable())

                ocr_results.append({
                    "page_num": page_num,
                    "text": cleaned_text,
                    "raw_text": text,
                    "confidence": float(data['conf'][i]) / 100,  # Convert to 0-1 range
                    "position": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w),
                        "y2": float(y + h)
                    }
                })

        doc.close()
        print(f"✅ Extracted {len(ocr_results)} text elements using OCR.")
        return ocr_results

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3):
        """Matches fields using AI and fills them immediately across multiple pages, ensuring OCR text is mapped to UUIDs properly."""

        backup_pdf = f"{pdf_path}.backup"
        shutil.copy2(pdf_path, backup_pdf)
        print(f"Created backup of original PDF: {backup_pdf}")

        print(json_data)

        pdf_fields = await self.extract_pdf_fields(pdf_path)
        ocr_text_elements = await self.extract_ocr_text(pdf_path)
        flat_json = self.flatten_json(json_data)
        field_context = await self.analyze_field_context(pdf_fields, ocr_text_elements)

        state = ""
        # Print available JSON fields for debugging
        print("Available JSON fields:")
        for key in flat_json.keys():
            print(key, flat_json[key])
            if key == "State.stateFullDesc" or key == "data.State.stateFullDesc" or key == "data.orderDetails.strapiOrderFormJson.State.stateFullDesc":
                print(flat_json[key])
                state = flat_json[key]

                print(state)

        if state == "Pennsylvania":
            print("Running 1")
            prompt = FIELD_MATCHING_PROMPT_UPDATED4.format(
                json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
                pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in pdf_fields.items()], indent=2,
                                      cls=NumpyEncoder),
                ocr_elements=json.dumps(ocr_text_elements, indent=2, cls=NumpyEncoder),
                field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
            )
        elif state == "Michigan":
            print("Running 2")
            prompt = FIELD_MATCHING_PROMPT_UPDATED.format(
                json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
                pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in pdf_fields.items()], indent=2,
                                      cls=NumpyEncoder),
                ocr_elements=json.dumps(ocr_text_elements, indent=2, cls=NumpyEncoder),
                field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
            )
        else:
            print("Running 3")
            prompt = FIELD_MATCHING_PROMPT_UPDATED1.format(
                json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
                pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in pdf_fields.items()], indent=2,
                                      cls=NumpyEncoder),
                ocr_elements=json.dumps(ocr_text_elements, indent=2, cls=NumpyEncoder),
                field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
            )

        matches, ocr_matches = [], []
        for attempt in range(max_retries):
            response = await self.agent.run(prompt)
            print(response.data)
            result = self.parse_ai_response(response.data)

            if result:
                matches = result.get("field_matches", [])
                ocr_matches = result.get("ocr_matches", [])
                if matches or ocr_matches:
                    break

            print(f"Attempt {attempt + 1}/{max_retries} failed to get valid matches. Retrying...")

        if not matches and not ocr_matches:
            print("⚠️ No valid field matches were found after all attempts.")
            return False

        temp_output = f"{output_pdf}.temp"
        shutil.copy2(pdf_path, temp_output)

        try:
            print("Filling form fields and OCR-detected fields together with UUID-based matching...")
            combined_matches = matches + [
                FieldMatch(
                    json_field=m.json_field,
                    pdf_field=m.pdf_field,  # Ensuring OCR text maps correctly to UUID
                    confidence=m.confidence,
                    suggested_value=m.suggested_value,
                    reasoning=m.reasoning
                ) for m in ocr_matches
            ]

            success = self.fill_pdf_immediately(temp_output, combined_matches, pdf_fields)

            if not success:
                print("⚠️ Some fields may not have been filled correctly.")

            if not success:
                print("⚠️ Some fields may not have been filled correctly.")
        except Exception as e:
            print(f"❌ Error during filling: {e}")
            return False

        try:
            self.finalize_pdf(temp_output, output_pdf)
            print(f"✅ Finalized PDF saved to: {output_pdf}")
            return self.verify_pdf_filled(output_pdf)
        except Exception as e:
            print(f"❌ Error during finalization: {e}")
            print("Trying alternative finalization method...")

            try:
                shutil.copy2(temp_output, output_pdf)
                print(f"✅ Alternative save successful: {output_pdf}")
                return self.verify_pdf_filled(output_pdf)
            except Exception as e2:
                print(f"❌ Alternative save also failed: {e2}")
                return False

    async def analyze_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                                    ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze context around form fields using AI to improve field understanding."""
        field_context = []

        # Prepare context data for AI analysis
        context_data = {
            "pdf_fields": [
                {
                    "field_name": field_name,
                    "page": field_info["page_num"] + 1,
                    "rect": field_info["rect"],
                    "type": field_info["type"],
                    "is_readonly": field_info["is_readonly"]
                } for field_name, field_info in pdf_fields.items()
            ],
            "ocr_elements": [
                {
                    "text": elem["text"],
                    "page_num": elem["page_num"] + 1,
                    "confidence": elem["confidence"],
                    "position": elem["position"]
                } for elem in ocr_elements
            ]
        }

        # Create a prompt for AI-based context analysis
        prompt = f"""
        Analyze the context of PDF form fields based on the following data:

        PDF Form Fields:
        {json.dumps(context_data['pdf_fields'], indent=2)}

        OCR Text Elements:
        {json.dumps(context_data['ocr_elements'], indent=2)}

        For each form field, identify and rank the most relevant nearby text elements 
        that might provide context or hints about the field's purpose. Consider:
        - Proximity of text to the field
        - Semantic relevance
        - Text confidence
        - Position relative to the field (left, above, right, below)

        Respond ONLY with a valid JSON in the following structure:
        {{
            "field_contexts": [
                {{
                    "field_name": "string",
                    "page": int,
                    "nearby_text": [
                        {{
                            "text": "string",
                            "position": "string",
                            "relevance_score": 0.0
                        }}
                    ]
                }}
            ]
        }}

        Important: Ensure the JSON is well-formed and can be parsed directly.
        """

        try:
            # Use the existing agent to analyze field context
            response = await self.agent.run(prompt)

            # Robust parsing with multiple fallback mechanisms
            cleaned_response = response.data.strip()

            # Remove any markdown code block markers
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            try:
                # First attempt: direct JSON parsing
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Second attempt: use ast for more lenient parsing
                import ast
                try:
                    result = ast.literal_eval(cleaned_response)
                except:
                    print("⚠️ Failed to parse AI response for field context.")
                    return self._fallback_field_context(pdf_fields, ocr_elements)

            # Validate the parsed result
            field_context = result.get("field_contexts", [])

            if not field_context:
                print("⚠️ No field contexts found. Falling back to default method.")
                field_context = self._fallback_field_context(pdf_fields, ocr_elements)

        except Exception as e:
            print(f"❌ Error in AI-based field context analysis: {e}")
            field_context = self._fallback_field_context(pdf_fields, ocr_elements)

        return field_context

    def _fallback_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                                ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        field_context = []

        for field_name, field_info in pdf_fields.items():
            page_num = field_info["page_num"]
            rect = field_info["rect"]

            nearby_text = []
            for ocr_elem in ocr_elements:
                if ocr_elem["page_num"] != page_num:
                    continue

                ocr_pos = ocr_elem["position"]

                distance_x = abs(rect[0] - ocr_pos["x2"])
                distance_y = abs(rect[1] - ocr_pos["y2"])

                if (ocr_pos["x2"] < rect[0] and distance_x < 200 and abs(rect[1] - ocr_pos["y1"]) < 30) or \
                        (ocr_pos["y2"] < rect[1] and distance_y < 50 and abs(rect[0] - ocr_pos["x1"]) < 200):
                    nearby_text.append({
                        "text": ocr_elem["text"],
                        "distance_x": distance_x,
                        "distance_y": distance_y,
                        "position": "left" if ocr_pos["x2"] < rect[0] else "above"
                    })

            nearby_text.sort(key=lambda x: x["distance_x"] + x["distance_y"])

            field_context.append({
                "field_name": field_name,
                "page": page_num + 1,
                "nearby_text": nearby_text[:3]
            })

        return field_context

    def parse_ai_response(self, response_text: str) -> Dict[str, List]:
        """Parses AI response and extracts valid JSON matches for both form fields and OCR text."""
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'(\{[\s\S]*\})'
        ]

        for pattern in json_patterns:
            json_match = re.search(pattern, response_text)
            if json_match:
                response_text = json_match.group(1)
                break

        response_text = response_text.strip()

        try:
            data = json.loads(response_text)
            result = {
                "field_matches": [],
                "ocr_matches": []
            }

            for match in data.get("matches", []):
                match.setdefault("confidence", 1.0)
                match.setdefault("reasoning", "No reasoning provided.")

                try:
                    validated_match = FieldMatch(**match)
                    result["field_matches"].append(validated_match)
                except Exception as e:
                    print(f"⚠️ Skipping malformed field match: {match} | Error: {e}")

            for match in data.get("ocr_matches", []):
                match.setdefault("confidence", 1.0)
                match.setdefault("reasoning", "No reasoning provided.")
                match.setdefault("page_num", 0)
                match.setdefault("x1", 100)
                match.setdefault("y1", 100)
                match.setdefault("x2", 300)
                match.setdefault("y2", 120)

                try:
                    validated_match = OCRFieldMatch(**match)
                    result["ocr_matches"].append(validated_match)
                except Exception as e:
                    print(f"⚠️ Skipping malformed OCR match: {match} | Error: {e}")

            return result
        except json.JSONDecodeError as e:
            print(f"❌ AI returned invalid JSON: {e}")
            print(f"Failed text: {response_text[:100]}...")
            return {}

    def fill_pdf_immediately(self, output_pdf: str, matches: List[FieldMatch],
                             pdf_fields: Dict[str, Dict[str, Any]]) -> bool:
        """Fills PDF form fields using PyMuPDF (fitz) with improved handling of readonly fields."""
        doc = fitz.open(output_pdf)
        filled_fields = []

        updates = []
        for match in matches:
            if match.pdf_field and match.suggested_value is not None:
                field_info = pdf_fields.get(match.pdf_field)

                if not field_info:
                    print(f"⚠️ Field '{match.pdf_field}' not found in PDF")
                    continue

                if field_info["is_readonly"]:
                    print(f"⚠️ Skipping readonly field '{match.pdf_field}' - will handle via OCR")
                    continue

                page_num = field_info["page_num"]
                updates.append((page_num, match.pdf_field, match.suggested_value))

        for page_num, field_name, value in updates:
            page = doc[page_num]
            for widget in page.widgets():
                if widget.field_name == field_name:
                    print(f"✍️ Filling: '{value}' → '{field_name}' (Page {page_num + 1})")
                    try:
                        widget.field_value = str(value)
                        widget.update()
                        filled_fields.append(field_name)
                    except Exception as e:
                        print(f"⚠️ Error filling {field_name}: {e}")
                    break

        try:
            # Remove garbage and incremental parameters to fix the error
            doc.save(output_pdf, deflate=True, clean=True)
            print(f"✅ Saved PDF with {len(filled_fields)} filled fields")
            doc.close()
            return len(filled_fields) > 0
        except Exception as e:
            print(f"❌ Error saving PDF: {e}")

            try:
                temp_path = f"{output_pdf}.tmp"
                doc.save(temp_path, deflate=True, clean=True)
                doc.close()
                shutil.move(temp_path, output_pdf)
                print(f"✅ Saved PDF using alternative method")
                return len(filled_fields) > 0
            except Exception as e2:
                print(f"❌ Alternative save also failed: {e2}")
                doc.close()
                return False

    def finalize_pdf(self, input_pdf: str, output_pdf: str) -> None:
        """Finalizes the PDF using PyPDF to avoid incremental save issues."""
        try:
            reader = PdfReader(input_pdf)
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            if reader.get_fields():
                writer.clone_reader_document_root(reader)

            with open(output_pdf, "wb") as f:
                writer.write(f)
        except Exception as e:
            print(f"❌ Error in finalize_pdf: {e}")
            # Simply copy the file as a fallback
            shutil.copy2(input_pdf, output_pdf)
            print(f"✅ Used direct file copy as fallback for finalization")

    def verify_pdf_filled(self, pdf_path: str) -> bool:
        """Verifies that the PDF has been filled correctly or has annotations."""
        try:
            reader = PdfReader(pdf_path)
            fields = reader.get_fields()

            filled_fields = {}
            if fields:
                filled_fields = {k: v.get("/V") for k, v in fields.items() if v.get("/V")}
                print(f"✅ Found {len(filled_fields)} filled form fields")

            doc = fitz.open(pdf_path)
            annotation_count = 0

            for page in doc:
                annotations = list(page.annots())
                annotation_count += len(annotations)

            doc.close()
            print(f"✅ Found {annotation_count} annotations in the PDF")

            return bool(filled_fields) or annotation_count > 0

        except Exception as e:
            print(f"❌ Error verifying PDF: {e}")
            return False

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
    template_pdf = "D:\\demo\\Services\\WisconsinCorp.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
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