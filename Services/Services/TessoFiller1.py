import asyncio
import json
import os
import re
import shutil
import io
from typing import Dict, Any, List, Tuple

import fitz
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator

from Common.constants import *

API_KEYS = {
    "field_matcher": API_KEY_3,
    "vision": API_KEY_3,  # Add vision API key (can be the same as field_matcher)
}

# Configure Gemini API
genai.configure(api_key=API_KEYS["vision"])


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
    pdf_field: str
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
    pdf_field: str
    suggested_value: Any
    reasoning: str

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)


class PageByPageFormFiller:
    def __init__(self):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["field_matcher"]),
            system_prompt="You are an expert at mapping PDF fields to JSON keys and filling them immediately."
        )

        self.matched_fields = {}

    async def extract_pdf_fields_for_page(self, doc: fitz.Document, page_num: int) -> Dict[str, Dict[str, Any]]:
        """Extracts fillable fields from a specific page of a PDF."""
        print(f"🔍 Extracting fillable fields from page {page_num + 1}...")
        fields = {}

        page = doc[page_num]
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

        print(f"✅ Extracted {len(fields)} fields from page {page_num + 1}")
        for field, info in fields.items():
            readonly_status = "READ-ONLY" if info["is_readonly"] else "EDITABLE"
            print(f" - Field: '{field}' [{readonly_status}]")

        return fields

    async def extract_ocr_for_page(self, doc: fitz.Document, page_num: int, dpi=300) -> List[Dict[str, Any]]:
        """Extract OCR text for a single page using Gemini Vision."""
        print(f"🔍 Extracting text using Gemini Vision for page {page_num + 1}...")

        page = doc[page_num]
        ocr_results = []

        try:
            # Get page image
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Process with Gemini Vision
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([
                "Extract all the text from this image, maintaining the structure and layout information. Also identify form fields and their positions. Return the following JSON structure: {\"extracted_text\": \"full text\", \"form_fields\": [{\"label\": \"field label\", \"position\": {\"x1\": float, \"y1\": float, \"x2\": float, \"y2\": float}}]}",
                Image.open(io.BytesIO(img_byte_arr))
            ])

            extracted_text = response.text.strip()

            # Try to parse JSON from the response
            try:
                json_data = json.loads(extracted_text)
                main_text = json_data.get("extracted_text", "")
                form_fields = json_data.get("form_fields", [])

                # Add main text
                if main_text:
                    ocr_results.append({
                        "text": main_text,
                        "page_num": page_num,
                        "confidence": 0.9,
                        "position": {
                            "x1": 0.0,
                            "y1": 0.0,
                            "x2": float(page.rect.width),
                            "y2": float(page.rect.height)
                        }
                    })

                # Add form fields
                for field in form_fields:
                    label = field.get("label", "")
                    position = field.get("position", {})
                    if label:
                        ocr_results.append({
                            "text": label,
                            "page_num": page_num,
                            "confidence": 0.95,
                            "position": {
                                "x1": float(position.get("x1", 0)),
                                "y1": float(position.get("y1", 0)),
                                "x2": float(position.get("x2", 100)),
                                "y2": float(position.get("y2", 20))
                            }
                        })
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                ocr_results.append({
                    "text": extracted_text,
                    "page_num": page_num,
                    "confidence": 0.9,
                    "position": {
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": float(page.rect.width),
                        "y2": float(page.rect.height)
                    }
                })

            print(f"✅ Extracted text from page {page_num + 1}")

        except Exception as e:
            print(f"❌ Error extracting OCR for page {page_num + 1}: {str(e)}")
            # Fallback to simple text extraction
            text = page.get_text()
            if text.strip():
                ocr_results.append({
                    "text": text.strip(),
                    "page_num": page_num,
                    "confidence": 0.7,
                    "position": {
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": float(page.rect.width),
                        "y2": float(page.rect.height)
                    }
                })
                print(f"✅ Extracted text from page {page_num + 1} using fallback method")

        return ocr_results

    async def process_page(self, doc: fitz.Document, page_num: int, flat_json: Dict[str, Any], output_pdf: str,
                           max_retries: int = 3) -> bool:
        """Process a single page: extract fields, extract OCR data, match fields with JSON, and fill fields."""
        print(f"\n=== Processing Page {page_num + 1} ===")

        # Extract fields for this page
        page_fields = await self.extract_pdf_fields_for_page(doc, page_num)
        if not page_fields:
            print(f"No form fields found on page {page_num + 1}, skipping...")
            return True

        # Extract OCR data for this page
        page_ocr = await self.extract_ocr_for_page(doc, page_num)

        # Analyze field context
        field_context = self.analyze_field_context(page_fields, page_ocr)

        # Create prompt for field matching
        prompt = FIELD_MATCHING_PROMPT_UPDATED.format(
            json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
            pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in page_fields.items()], indent=2, cls=NumpyEncoder),
            ocr_elements=json.dumps(page_ocr, indent=2, cls=NumpyEncoder),
            field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
        )

        # Match fields with JSON using AI
        matches, ocr_matches = [], []
        for attempt in range(max_retries):
            response = await self.agent.run(prompt)
            result = self.parse_ai_response(response.data)

            if result:
                matches = result.get("field_matches", [])
                ocr_matches = result.get("ocr_matches", [])
                if matches or ocr_matches:
                    break

            print(
                f"Attempt {attempt + 1}/{max_retries} failed to get valid matches for page {page_num + 1}. Retrying...")

        if not matches and not ocr_matches:
            print(f"⚠️ No valid field matches were found for page {page_num + 1} after all attempts.")
            return True  # Continue with next page

        # Fill form fields
        print(f"Filling form fields for page {page_num + 1}...")
        combined_matches = matches + [
            FieldMatch(
                json_field=m.json_field,
                pdf_field=m.pdf_field,
                confidence=m.confidence,
                suggested_value=m.suggested_value,
                reasoning=m.reasoning
            ) for m in ocr_matches
        ]

        success = self.fill_pdf_immediately(output_pdf, combined_matches, page_fields)
        if not success:
            print(f"⚠️ Some fields may not have been filled correctly on page {page_num + 1}.")

        # Fill OCR-detected fields if needed
        if ocr_matches:
            ocr_success = self.fill_ocr_fields(output_pdf, ocr_matches, page_ocr)
            if not ocr_success:
                print(f"⚠️ Some OCR fields may not have been filled correctly on page {page_num + 1}.")

        return True

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3):
        """Matches fields using AI and fills them immediately page by page."""

        backup_pdf = f"{pdf_path}.backup"
        shutil.copy2(pdf_path, backup_pdf)
        print(f"Created backup of original PDF: {backup_pdf}")

        # Flatten JSON for easier matching
        flat_json = self.flatten_json(json_data)

        # Copy to temp output path first
        temp_output = f"{output_pdf}.temp"
        shutil.copy2(pdf_path, temp_output)

        try:
            # Open document
            doc = fitz.open(pdf_path)
            page_count = len(doc)

            # Process each page sequentially
            for page_num in range(page_count):
                success = await self.process_page(doc, page_num, flat_json, temp_output, max_retries)
                if not success:
                    print(f"⚠️ Failed to process page {page_num + 1}")

            doc.close()

            # Finalize the PDF
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

        except Exception as e:
            print(f"❌ Error during filling: {e}")
            return False

    def analyze_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                              ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze context around form fields to improve field understanding."""
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

    def create_label_field_map(self, ocr_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create a map of potential form labels to nearby fields."""
        label_map = {}

        by_page = {}
        for elem in ocr_elements:
            page = elem["page_num"]
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(elem)

        for page, elements in by_page.items():

            elements.sort(key=lambda x: x["position"]["y1"])

            for i, elem in enumerate(elements):
                if len(elem["text"]) > 30:
                    continue

                potential_fields = []
                for j, other in enumerate(elements):
                    if i == j:
                        continue

                    if (other["position"]["x1"] > elem["position"]["x2"] and
                            abs(other["position"]["y1"] - elem["position"]["y1"]) < 20):
                        potential_fields.append({
                            "text": other["text"],
                            "position": other["position"],
                            "relation": "right",
                            "distance": other["position"]["x1"] - elem["position"]["x2"]
                        })

                    elif (other["position"]["y1"] > elem["position"]["y2"] and
                          abs(other["position"]["x1"] - elem["position"]["x1"]) < 40 and
                          other["position"]["y1"] - elem["position"]["y2"] < 40):
                        potential_fields.append({
                            "text": other["text"],
                            "position": other["position"],
                            "relation": "below",
                            "distance": other["position"]["y1"] - elem["position"]["y2"]
                        })

                if potential_fields:
                    potential_fields.sort(key=lambda x: x["distance"])
                    label_map[elem["text"]] = potential_fields[:2]

        return label_map

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

    def fill_ocr_fields(self, pdf_path: str, ocr_matches: List[OCRFieldMatch],
                        ocr_elements: List[Dict[str, Any]]) -> bool:
        """Fills OCR-detected areas with text for readonly fields."""
        doc = fitz.open(pdf_path)
        annotations_added = 0

        for match in ocr_matches:
            if match.suggested_value is not None:
                try:
                    page = doc[match.page_num]

                    position = None
                    if match.ocr_text:
                        position = self.find_text_position(match.ocr_text, ocr_elements, match.page_num)

                    if position:
                        x1, y1, x2, y2 = position["x1"], position["y1"], position["x2"], position["y2"]

                        x1 = x2 + 10
                        x2 = x1 + 150

                        y2 = y1 + (y2 - y1)
                    else:
                        x1, y1, x2, y2 = match.x1, match.y1, match.x2, match.y2

                    rect = fitz.Rect(x1, y1, x2, y2)

                    print(
                        f"✍️ Filling OCR field: '{match.suggested_value}' → near '{match.ocr_text}' (Page {match.page_num + 1})")

                    annotation_added = False

                    try:
                        annot = page.add_freetext_annot(
                            rect=rect,
                            text=str(match.suggested_value),
                            fontsize=10,
                            fill_color=(0.95, 0.95, 0.95),
                            text_color=(0, 0, 0)
                        )
                        annotations_added += 1
                        annotation_added = True
                    except Exception as e1:
                        print(f"⚠️ Free text annotation failed: {e1}")

                    if not annotation_added:
                        try:
                            page.draw_rect(rect, color=(0.95, 0.95, 0.95), fill=(0.95, 0.95, 0.95))

                            page.insert_text(
                                point=(x1 + 2, y1 + 10),
                                text=str(match.suggested_value),
                                fontsize=10
                            )
                            annotations_added += 1
                            annotation_added = True
                        except Exception as e2:
                            print(f"⚠️ Text insertion failed: {e2}")

                    if not annotation_added:
                        try:
                            annot = page.add_text_annot(
                                point=(x1, y1),
                                text=str(match.suggested_value)
                            )
                            annotations_added += 1
                        except Exception as e3:
                            print(f"⚠️ All text methods failed: {e3}")
                except Exception as e:
                    print(f"⚠️ Error processing OCR match: {e}")

        if annotations_added > 0:
            try:
                # Remove incremental parameter to fix the error
                doc.save(pdf_path, deflate=True, clean=True)
                print(f"✅ Added {annotations_added} OCR text fields")
                doc.close()
                return True
            except Exception as e:
                print(f"❌ Error saving PDF with OCR annotations: {e}")
                try:
                    temp_path = f"{pdf_path}.temp"
                    doc.save(temp_path)
                    doc.close()
                    shutil.move(temp_path, pdf_path)
                    print(f"✅ Saved OCR annotations using alternative method")
                    return True
                except Exception as e2:
                    print(f"❌ Alternative save method also failed: {e2}")
                    doc.close()
                    return False
        else:
            doc.close()
            return False

    def find_text_position(self, text: str, ocr_elements: List[Dict[str, Any]], page_num: int) -> Dict[str, float]:
        """Find the position of a text element in the OCR results with improved fuzzy matching."""
        if not text or not ocr_elements:
            return None

        search_text = text.strip().lower()

        for element in ocr_elements:
            if element["page_num"] == page_num and element["text"].strip().lower() == search_text:
                return element["position"]

        for element in ocr_elements:
            if element["page_num"] == page_num and search_text in element["text"].strip().lower():
                return element["position"]

        best_match = None
        best_ratio = 0.7

        for element in ocr_elements:
            if element["page_num"] == page_num:
                element_text = element["text"].strip().lower()

                ratio = SequenceMatcher(None, search_text, element_text).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = element["position"]

                words_in_search = set(search_text.split())
                words_in_element = set(element_text.split())
                common_words = words_in_search.intersection(words_in_element)

                if common_words:
                    word_ratio = len(common_words) / max(len(words_in_search), 1)
                    if word_ratio > 0.5 and word_ratio > best_ratio:
                        best_ratio = word_ratio
                        best_match = element["position"]

        return best_match

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
    form_filler = PageByPageFormFiller()
    template_pdf = "D:\\demo\\Services\\nevada.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
    output_pdf = "D:\\demo\\Services\\fill_smart6.pdf"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

    if success:
        print(f"✅ PDF successfully processed: {output_pdf}")
    else:
        print(f"❌ PDF processing failed. Please check the output file and logs.")


if __name__ == "__main__":
    asyncio.run(main())