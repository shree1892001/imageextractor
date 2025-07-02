import asyncio
import json
import os
import re
import shutil
from typing import Dict, Any, List, Tuple

import fitz
import numpy as np
import cv2
from paddleocr import PaddleOCR
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator

from Common.constants import *

API_KEYS = {
    "field_matcher": API_KEY_3,
    "context_analyzer": API_KEY_3,  # Using the same key for both tasks
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


class FieldContext(BaseModel):
    field_name: str
    page: int
    nearby_text: List[Dict[str, Any]]


class MultiAgentFormFiller:
    def __init__(self):
        self.field_matcher_agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["field_matcher"]),
            system_prompt="You are an expert at mapping PDF fields to JSON keys and filling them immediately."
        )

        self.context_analyzer_agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["context_analyzer"]),
            system_prompt="You are an expert at analyzing PDF form structures and identifying relationships between form labels and fields."
        )

        self.ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

        self.matched_fields = {}
        # Cache for AI responses to reduce redundant API calls
        self.ai_cache = {}

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
        """Extract text from PDF using OCR with position information."""
        print("🔍 Extracting text using OCR...")
        doc = fitz.open(pdf_path)
        ocr_results = []

        for page_num in range(len(doc)):
            print(f"Processing OCR for page {page_num + 1}/{len(doc)}...")
            pix = doc[page_num].get_pixmap(alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            results = self.ocr_reader.ocr(binary, cls=True)

            if not results[0]:
                _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                additional_results = self.ocr_reader.ocr(threshold, cls=True)

                if additional_results[0]:
                    results = additional_results

            if results[0]:
                unique_results = []
                seen_texts = set()

                for line in results[0]:
                    bbox, (text, prob) = line

                    text = text.strip().lower()
                    if text and text not in seen_texts and prob >= 0.4:
                        seen_texts.add(text)
                        unique_results.append((bbox, text, prob))

                for (bbox, text, prob) in unique_results:
                    if prob < 0.4 or not text.strip():
                        continue

                    # PaddleOCR format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[2]

                    cleaned_text = text.strip()
                    cleaned_text = ''.join(c for c in cleaned_text if c.isprintable())

                    ocr_results.append({
                        "page_num": page_num,
                        "text": cleaned_text,
                        "raw_text": text,
                        "confidence": float(prob),
                        "position": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
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

        pdf_fields = await self.extract_pdf_fields(pdf_path)
        ocr_text_elements = await self.extract_ocr_text(pdf_path)
        flat_json = self.flatten_json(json_data)

        # Use AI to analyze field context
        field_context = await self.ai_analyze_field_context(pdf_fields, ocr_text_elements)

        # Print available JSON fields for debugging
        print("Available JSON fields:")
        for key in flat_json.keys():
            print(f" - {key}: {flat_json[key]}")

        prompt = FIELD_MATCHING_PROMPT_UPDATED.format(
            json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
            pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in pdf_fields.items()], indent=2, cls=NumpyEncoder),
            ocr_elements=json.dumps(ocr_text_elements, indent=2, cls=NumpyEncoder),
            field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
        )

        matches, ocr_matches = [], []
        for attempt in range(max_retries):
            response = await self.field_matcher_agent.run(prompt)
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
                    pdf_field=m.pdf_field,
                    confidence=m.confidence,
                    suggested_value=m.suggested_value,
                    reasoning=m.reasoning
                ) for m in ocr_matches
            ]

            success = self.fill_pdf_immediately(temp_output, combined_matches, pdf_fields)

            # Fill OCR fields for readonly fields
            ocr_success = await self.fill_ocr_fields(temp_output, ocr_matches, ocr_text_elements)

            if not success and not ocr_success:
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

    def flatten_json(self, json_data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flattens nested JSON structure."""
        items = []
        for k, v in json_data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self.flatten_json(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self.flatten_json(item, f"{new_key}[{i}]", sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    async def ai_analyze_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                                       ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to analyze context around form fields."""
        print("🧠 Using AI to analyze field context...")

        # Create cache key based on field count and OCR element count
        cache_key = f"field_context_{len(pdf_fields)}_{len(ocr_elements)}"
        if cache_key in self.ai_cache:
            print("📋 Using cached field context analysis")
            return self.ai_cache[cache_key]

        # Prepare data for AI analysis
        fields_with_position = []
        for field_name, field_info in pdf_fields.items():
            fields_with_position.append({
                "field_name": field_name,
                "page_num": field_info["page_num"],
                "rect": field_info["rect"],
                "is_readonly": field_info["is_readonly"]
            })

        # Sample OCR elements to reduce prompt size if there are too many
        sample_ocr_elements = ocr_elements
        if len(ocr_elements) > 200:  # Limit for context
            # Group by page and take samples from each page
            by_page = {}
            for elem in ocr_elements:
                page = elem["page_num"]
                if page not in by_page:
                    by_page[page] = []
                by_page[page].append(elem)

            sample_ocr_elements = []
            for page, elements in by_page.items():
                # Take up to 50 elements per page
                sample_size = min(50, len(elements))
                sampled = np.random.choice(elements, sample_size, replace=False).tolist()
                sample_ocr_elements.extend(sampled)

        # Create prompt for AI to analyze field context
        context_prompt = """
        You are an expert in analyzing PDF forms and identifying relationships between form labels and fields.

        Given the PDF fields and OCR text elements, analyze the context around each form field to determine:
        1. What nearby text is likely to be a label for the field
        2. The relative position of the label to the field (left, above, etc.)
        3. Any additional context that might help identify the field's purpose

        For each field, provide the most relevant nearby text elements that could help identify the field.

        PDF Form Fields:
        {pdf_fields}

        OCR Text Elements (sample):
        {ocr_elements}

        Return a structured output in the following format:
        ```json
        [
          {{
            "field_name": "field1",
            "page": 1,
            "nearby_text": [
              {{
                "text": "Label text",
                "position": "left",
                "confidence": 0.9
              }}
            ]
          }},
          ...
        ]
        ```
        """

        context_prompt = context_prompt.format(
            pdf_fields=json.dumps(fields_with_position, indent=2, cls=NumpyEncoder),
            ocr_elements=json.dumps(sample_ocr_elements, indent=2, cls=NumpyEncoder)
        )

        # Get AI response
        response = await self.context_analyzer_agent.run(context_prompt)

        # Parse AI response
        field_context = self.parse_field_context_response(response.data)

        if not field_context:
            print("⚠️ AI field context analysis failed, falling back to basic analysis")
            # Fallback to a simplified analysis
            field_context = self.analyze_field_context_fallback(pdf_fields, ocr_elements)

        # Cache the result
        self.ai_cache[cache_key] = field_context
        return field_context

    def parse_field_context_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the AI response for field context analysis."""
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'(\[[\s\S]*\])'
        ]

        for pattern in json_patterns:
            json_match = re.search(pattern, response_text)
            if json_match:
                response_text = json_match.group(1)
                break

        response_text = response_text.strip()

        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            print(f"❌ AI returned invalid JSON for field context analysis")
            return []

    def analyze_field_context_fallback(self, pdf_fields: Dict[str, Dict[str, Any]],
                                       ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback method for field context analysis if AI fails."""
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

    async def ai_create_label_field_map(self, ocr_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Use AI to create a map of potential form labels to nearby fields."""
        print("🧠 Using AI to analyze label-field relationships...")

        # Create cache key based on OCR element count
        cache_key = f"label_field_map_{len(ocr_elements)}"
        if cache_key in self.ai_cache:
            print("📋 Using cached label-field map")
            return self.ai_cache[cache_key]

        # Sample OCR elements if there are too many
        sample_ocr_elements = ocr_elements
        if len(ocr_elements) > 200:
            sample_ocr_elements = np.random.choice(ocr_elements, 200, replace=False).tolist()

        # Create prompt for AI to analyze label-field relationships
        prompt = """
        You are an expert in analyzing PDF forms and identifying relationships between form labels and fields.

        Given the OCR text elements from a PDF form, identify which elements are likely to be form labels 
        and which nearby elements are likely to be form fields that correspond to those labels.

        OCR Text Elements (sample):
        {ocr_elements}

        Return a structured mapping of labels to their corresponding fields in the following format:
        ```json
        {{
          "Label 1": [
            {{
              "text": "Field text",
              "position": "right",
              "distance": 20
            }}
          ],
          "Label 2": [
            {{
              "text": "Field text",
              "position": "below",
              "distance": 15
            }}
          ]
        }}
        ```

        Focus on identifying clear label-field relationships based on:
        1. Proximity (labels are typically close to their fields)
        2. Relative position (labels are typically to the left or above their fields)
        3. Formatting (labels often end with a colon or similar punctuation)
        """

        prompt = prompt.format(
            ocr_elements=json.dumps(sample_ocr_elements, indent=2, cls=NumpyEncoder)
        )

        # Get AI response
        response = await self.context_analyzer_agent.run(prompt)

        # Parse AI response
        label_field_map = self.parse_label_field_map_response(response.data)

        if not label_field_map:
            print("⚠️ AI label-field map analysis failed, falling back to basic analysis")
            # Fallback to a simplified analysis
            label_field_map = self.create_label_field_map_fallback(ocr_elements)

        # Cache the result
        self.ai_cache[cache_key] = label_field_map
        return label_field_map

    def parse_label_field_map_response(self, response_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the AI response for label-field map analysis."""
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
            if isinstance(data, dict):
                return data
            return {}
        except json.JSONDecodeError:
            print(f"❌ AI returned invalid JSON for label-field map analysis")
            return {}

    def create_label_field_map_fallback(self, ocr_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback method for creating label-field map if AI fails."""
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

    async def find_text_position_ai(self, text: str, ocr_elements: List[Dict[str, Any]], page_num: int) -> Dict[
        str, float]:
        """Use AI to find the position of text in OCR elements."""
        if not text or not ocr_elements:
            return None

        # Create cache key
        cache_key = f"text_position_{text}_{page_num}"
        if cache_key in self.ai_cache:
            return self.ai_cache[cache_key]

        page_elements = [e for e in ocr_elements if e["page_num"] == page_num]
        if not page_elements:
            return None

        prompt = """
        You are an expert in text matching and similarity comparison.

        Given a search text and a list of OCR text elements, find the element that best matches the search text.
        Consider exact matches, partial matches, substring matches, and semantic similarity.

        Search text: "{search_text}"

        OCR Elements:
        {ocr_elements}

        Return the index of the best matching element and your confidence score (0-1) in the following format:
        ```json
        {{
          "best_match_index": 0,
          "confidence": 0.9,
          "reasoning": "Exact match found."
        }}
        ```

        If no good match is found, return:
        ```json
        {{
          "best_match_index": null,
          "confidence": 0,
          "reasoning": "No suitable match found."
        }}
        ```
        """

        prompt = prompt.format(
            search_text=text,
            ocr_elements=json.dumps([
                {"index": i, "text": e["text"]} for i, e in enumerate(page_elements)
            ], indent=2)
        )

        response = await self.context_analyzer_agent.run(prompt)

        # Parse AI response
        match_result = self.parse_text_position_response(response.data)

        position = None
        if match_result and match_result.get("best_match_index") is not None:
            best_index = match_result["best_match_index"]
            if 0 <= best_index < len(page_elements):
                position = page_elements[best_index]["position"]

        # Fallback to traditional method if AI fails
        if not position:
            position = self.find_text_position(text, ocr_elements, page_num)

        # Cache the result
        if position:
            self.ai_cache[cache_key] = position

        return position

    def parse_text_position_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI response for text position finding."""
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
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"❌ AI returned invalid JSON for text position finding")
            return {}

    def find_text_position(self, text: str, ocr_elements: List[Dict[str, Any]], page_num: int) -> Dict[str, float]:
        """Traditional method to find text position based on string similarity."""
        page_elements = [e for e in ocr_elements if e["page_num"] == page_num]
        text = text.lower().strip()

        best_match = None
        best_ratio = 0

        for element in page_elements:
            ratio = SequenceMatcher(None, text, element["text"].lower()).ratio()
            if ratio > best_ratio and ratio > 0.7:  # Require at least 70% similarity
                best_ratio = ratio
                best_match = element

        return best_match["position"] if best_match else None

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

                if field_info.get("is_readonly", False):
                    print(f"⚠️ Field '{match.pdf_field}' is read-only, saving for OCR filling")
                    continue

                page = doc[field_info["page_num"]]
                field_type = field_info["type"]
                value = match.suggested_value

                # Process value based on field type
                if field_type == 4:  # Text field
                    if not isinstance(value, str):
                        value = str(value)
                    elif field_type == 2:  # Checkbox
                        if isinstance(value, bool):
                            value = value
                    elif isinstance(value, str):
                            value = str(value).lower() in ("yes", "true", "1", "x", "checked")
                    else:
                            value = bool(value)

                updates.append((page, match.pdf_field, value))

        # Apply all updates at once to avoid field reset issues
        for page, field_name, value in updates:
            # Fix: Convert widget generator to list before checking length
            widget_list = list(page.widgets(field_name))
            if widget_list:  # Check if list is not empty
                widget_list[0].field_value = value
                widget_list[0].update()
                filled_fields.append(field_name)
                print(f"✅ Filled '{field_name}' with value: {value}")

        doc.save(output_pdf, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()

        print(f"✅ Successfully filled {len(filled_fields)} fields")
        return len(filled_fields) > 0

    async def fill_ocr_fields(self, pdf_path: str, ocr_matches: List[OCRFieldMatch],
                              ocr_elements: List[Dict[str, Any]]) -> bool:
        """Fills fields identified through OCR by adding overlay text."""
        if not ocr_matches:
            return False

        doc = fitz.open(pdf_path)
        filled_count = 0

        for match in ocr_matches:
            if match.suggested_value is None:
                continue

            value = match.suggested_value
            if not isinstance(value, str):
                value = str(value)

            # Try to find the position using AI first
            position = await self.find_text_position_ai(match.ocr_text, ocr_elements, match.page_num)

            if position:
                # Use OCR position
                x1, y1 = position["x1"], position["y1"]
                x2, y2 = position["x2"], position["y2"]
            else:
                # Use provided coordinates as fallback
                x1, y1 = match.x1, match.y1
                x2, y2 = match.x2, match.y2

            # Adjust position to place text in a reasonable location
            text_x = x2 + 10  # Place text to the right of the label
            text_y = y1 + (y2 - y1) / 2  # Vertically center

            page = doc[match.page_num]

            # Add a white rectangle to cover any existing text
            rect = fitz.Rect(text_x, y1 - 2, text_x + 200, y2 + 2)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

            # Add the new text
            text_color = (0, 0, 0)  # Black
            font_size = min(12, (y2 - y1) * 0.8)  # Adjust font size based on field height
            page.insert_text((text_x, text_y), value, fontsize=font_size, color=text_color)

            filled_count += 1
            print(f"✅ Added OCR text for '{match.json_field}' with value: {value}")

        doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()

        print(f"✅ Successfully filled {filled_count} OCR-identified fields")
        return filled_count > 0

    def finalize_pdf(self, temp_pdf: str, output_pdf: str):
        """Finalizes the PDF by flattening form fields and cleaning up."""
        try:
            # First approach: Use pypdf to create a clean copy
            reader = PdfReader(temp_pdf)
            writer = PdfWriter()

            # Copy all pages
            for page in reader.pages:
                writer.add_page(page)

            # Update document info and encryption
            metadata = reader.metadata
            if metadata:
                writer.add_metadata(metadata)

            # Try to add needed permissions
            permissions = DictionaryObject({
                NameObject("/Print"): BooleanObject(True),
                NameObject("/Modify"): BooleanObject(True),
                NameObject("/Copy"): BooleanObject(True),
                NameObject("/AnnotForms"): BooleanObject(True)
            })

            # Get encryption algorithm from source if available
            encryption = reader.encrypt
            if encryption:
                writer._encrypt = encryption

            # Update form fields
            if reader.AcroForm:
                writer._root_object[NameObject('/AcroForm')] = reader.AcroForm.get_object()

            with open(output_pdf, "wb") as output_file:
                writer.write(output_file)

        except Exception as e:
            print(f"❌ Error in pypdf approach: {e}")
            print("Falling back to direct copy...")
            shutil.copy2(temp_pdf, output_pdf)

    def verify_pdf_filled(self, pdf_path: str) -> bool:
        """Verifies that the PDF has form fields filled."""
        doc = fitz.open(pdf_path)

        filled_field_count = 0
        field_count = 0

        for page in doc:
            for widget in page.widgets():
                field_count += 1
                if widget.field_value:
                    filled_field_count += 1

        fill_rate = filled_field_count / field_count if field_count > 0 else 0
        doc.close()

        print(f"📊 Field fill rate: {filled_field_count}/{field_count} ({fill_rate:.1%})")

        # Consider success if at least some fields are filled
        return filled_field_count > 0

    # Add a method to handle multi-page forms with complex field relationships
    async def analyze_field_relationships(self, pdf_fields: Dict[str, Dict[str, Any]],
                                          ocr_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to analyze relationships between fields across pages."""
        print("🧠 Using AI to analyze field relationships across pages...")

        # Create cache key based on field count
        cache_key = f"field_relationships_{len(pdf_fields)}"
        if cache_key in self.ai_cache:
            print("📋 Using cached field relationship analysis")
            return self.ai_cache[cache_key]

        # Group fields by page
        fields_by_page = {}
        for field_name, field_info in pdf_fields.items():
            page_num = field_info["page_num"]
            if page_num not in fields_by_page:
                fields_by_page[page_num] = []
            fields_by_page[page_num].append({
                "field_name": field_name,
                "rect": field_info["rect"],
                "type": field_info["type"]
            })

        # Create prompt for AI to analyze field relationships
        prompt = """
        You are an expert in analyzing PDF forms and identifying relationships between fields.

        Given the PDF form fields grouped by page, analyze potential relationships between fields, such as:
        1. Which fields might contain related information (e.g., first name and last name)
        2. Which fields might be part of the same section or group
        3. Whether there are repeating patterns of fields across multiple pages
        4. Whether there are dependencies between fields (e.g., "If Other, please specify")

        PDF Form Fields by Page:
        {fields_by_page}

        Return a structured analysis in the following format:
        ```json
        {{
          "field_groups": [
            {{
              "group_name": "Personal Information",
              "fields": ["FirstName", "LastName", "DOB"],
              "confidence": 0.9
            }}
          ],
          "field_dependencies": [
            {{
              "primary_field": "PaymentMethod",
              "dependent_fields": ["CreditCardNumber", "ExpiryDate"],
              "dependency_type": "conditional",
              "condition": "PaymentMethod == 'Credit Card'",
              "confidence": 0.8
            }}
          ],
          "repeating_patterns": [
            {{
              "pattern_name": "Address Fields",
              "fields": ["Street", "City", "State", "ZIP"],
              "occurrences": [0, 2],
              "confidence": 0.85
            }}
          ]
        }}
        ```
        """

        prompt = prompt.format(
            fields_by_page=json.dumps(fields_by_page, indent=2, cls=NumpyEncoder)
        )

        # Get AI response
        response = await self.context_analyzer_agent.run(prompt)

        # Parse AI response
        relationships = self.parse_field_relationships_response(response.data)

        if not relationships:
            print("⚠️ AI field relationship analysis failed, using empty result")
            relationships = {
                "field_groups": [],
                "field_dependencies": [],
                "repeating_patterns": []
            }

        # Cache the result
        self.ai_cache[cache_key] = relationships
        return relationships

    def parse_field_relationships_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI response for field relationship analysis."""
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
            if isinstance(data, dict):
                return data
            return {}
        except json.JSONDecodeError:
            print(f"❌ AI returned invalid JSON for field relationship analysis")
            return {}

    # Add main function to use the multi-agent form filler
async def main():
    form_filler = MultiAgentFormFiller()
    template_pdf = "D:\\demo\\Services\\WisconsinLLC.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
    output_pdf = "D:\\demo\\Services\\fill_smart5.pdf"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

    if success:
        print(f"✅ PDF successfully processed: {output_pdf}")
    else:
        print(f"❌ PDF processing failed. Please check the output file and logs.")


if __name__ == "__main__":
    asyncio.run(main())