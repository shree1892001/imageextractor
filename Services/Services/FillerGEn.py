import asyncio
import json
import os
import re
import shutil
import logging
import traceback
from typing import Dict, Any, List, Tuple
from datetime import datetime

import fitz
import numpy as np
import cv2
import easyocr
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator

from Common.constants import *

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"form_filler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FormFiller")

API_KEYS = {
    "field_matcher": API_KEY_3,
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
    suggested_value: Any
    reasoning: str

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)


class FormFillerError(Exception):
    """Base exception for form filler operations."""
    pass


class PDFExtractionError(FormFillerError):
    """Raised when there's an issue extracting data from PDF."""
    pass


class AIProcessingError(FormFillerError):
    """Raised when there's an issue with AI processing."""
    pass


class PDFWriteError(FormFillerError):
    """Raised when there's an issue writing to PDF."""
    pass


class MultiAgentFormFiller:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        try:
            self.agent = Agent(
                model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["field_matcher"]),
                system_prompt="You are an expert at mapping PDF fields to JSON keys and filling them immediately."
            )
            logger.info("AI agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI agent: {e}")
            raise AIProcessingError(f"AI agent initialization failed: {str(e)}")

        try:
            logger.info("Initializing OCR reader...")
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise FormFillerError(f"OCR reader initialization failed: {str(e)}")

        self.matched_fields = {}

    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        """Extracts all fillable fields from a multi-page PDF with additional metadata."""
        logger.info(f"🔍 Extracting fillable fields from: {pdf_path}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
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

            logger.info(f"✅ Extracted {len(fields)} fields across {len(doc)} pages.")

            if self.debug_mode:
                for field, info in fields.items():
                    readonly_status = "READ-ONLY" if info["is_readonly"] else "EDITABLE"
                    logger.debug(f" - Field: '{field}' (Page {info['page_num'] + 1}) [{readonly_status}]")

            doc.close()
            return fields

        except Exception as e:
            error_msg = f"Failed to extract PDF fields: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PDFExtractionError(error_msg)

    async def extract_ocr_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using OCR with position information."""
        logger.info(f"🔍 Extracting text using OCR from: {pdf_path}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            doc = fitz.open(pdf_path)
            ocr_results = []

            for page_num in range(len(doc)):
                logger.info(f"Processing OCR for page {page_num + 1}/{len(doc)}...")
                try:
                    pix = doc[page_num].get_pixmap(alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2)

                    results = self.ocr_reader.readtext(binary)

                    if len(results) < 10:
                        logger.info(f"First OCR pass yielded fewer than 10 results, trying alternative threshold...")
                        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                        more_results = self.ocr_reader.readtext(threshold)
                        logger.info(f"Alternative OCR found {len(more_results)} additional results")
                        results.extend(more_results)

                        unique_results = []
                        seen_texts = set()
                        for item in results:
                            text = item[1].strip().lower()
                            if text and text not in seen_texts:
                                seen_texts.add(text)
                                unique_results.append(item)
                        results = unique_results
                        logger.info(f"Deduplicated to {len(results)} unique OCR results")

                    page_ocr_count = 0
                    for (bbox, text, prob) in results:
                        if prob < 0.4 or not text.strip():
                            continue

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
                        page_ocr_count += 1

                    logger.info(f"Extracted {page_ocr_count} text elements from page {page_num + 1}")

                except Exception as page_e:
                    logger.error(f"Error processing OCR for page {page_num + 1}: {page_e}")
                    logger.error(traceback.format_exc())
                    # Continue with other pages instead of failing completely

            doc.close()
            logger.info(f"✅ Total OCR extraction: {len(ocr_results)} text elements")

            if self.debug_mode and len(ocr_results) > 0:
                sample_size = min(5, len(ocr_results))
                logger.debug(f"Sample of {sample_size} OCR results:")
                for i in range(sample_size):
                    logger.debug(f"  {i + 1}. '{ocr_results[i]['text']}' (conf: {ocr_results[i]['confidence']:.2f})")

            return ocr_results

        except Exception as e:
            error_msg = f"Failed to extract OCR text: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PDFExtractionError(error_msg)

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3):
        """Matches fields using AI and fills them immediately across multiple pages, ensuring OCR text is mapped to UUIDs properly."""
        logger.info(f"Starting field matching and filling process for: {pdf_path}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create backup of original PDF
        try:
            backup_pdf = f"{pdf_path}.backup"
            shutil.copy2(pdf_path, backup_pdf)
            logger.info(f"Created backup of original PDF: {backup_pdf}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}. Proceeding without backup.")

        try:
            # Extract form fields and OCR text
            pdf_fields = await self.extract_pdf_fields(pdf_path)
            ocr_text_elements = await self.extract_ocr_text(pdf_path)
            flat_json = self.flatten_json(json_data)
            field_context = self.analyze_field_context(pdf_fields, ocr_text_elements)

            # Log available JSON fields for debugging
            if self.debug_mode:
                logger.debug("Available JSON fields:")
                for key, value in flat_json.items():
                    logger.debug(f" - {key}: {value}")

            prompt = FIELD_MATCHING_PROMPT_UPDATED.format(
                json_data=json.dumps(flat_json, indent=2, cls=NumpyEncoder),
                pdf_fields=json.dumps([{"uuid": k, "info": v} for k, v in pdf_fields.items()], indent=2,
                                      cls=NumpyEncoder),
                ocr_elements=json.dumps(ocr_text_elements, indent=2, cls=NumpyEncoder),
                field_context=json.dumps(field_context, indent=2, cls=NumpyEncoder)
            )

            # Save prompt for debugging if needed
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(log_file), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, "last_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
                logger.debug(f"Saved prompt to {os.path.join(debug_dir, 'last_prompt.txt')}")

            # Get matches from AI with retries
            matches, ocr_matches = [], []
            for attempt in range(max_retries):
                try:
                    logger.info(f"AI matching attempt {attempt + 1}/{max_retries}")
                    response = await self.agent.run(prompt)

                    if self.debug_mode:
                        # Save AI response for debugging
                        with open(os.path.join(debug_dir, f"ai_response_{attempt + 1}.txt"), "w",
                                  encoding="utf-8") as f:
                            f.write(response.data)

                    result = self.parse_ai_response(response.data)

                    if result:
                        matches = result.get("field_matches", [])
                        ocr_matches = result.get("ocr_matches", [])
                        if matches or ocr_matches:
                            logger.info(
                                f"Successfully got {len(matches)} field matches and {len(ocr_matches)} OCR matches")
                            break

                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to get valid matches.")

                except Exception as e:
                    logger.error(f"Error during AI processing attempt {attempt + 1}: {e}")
                    logger.error(traceback.format_exc())
                    # Sleep before retry
                    await asyncio.sleep(2)

            if not matches and not ocr_matches:
                error_msg = "No valid field matches were found after all attempts."
                logger.error(error_msg)
                raise AIProcessingError(error_msg)

            # Create temporary output file
            temp_output = f"{output_pdf}.temp"
            try:
                shutil.copy2(pdf_path, temp_output)
                logger.info(f"Created temporary working copy: {temp_output}")
            except Exception as e:
                error_msg = f"Failed to create temporary file: {e}"
                logger.error(error_msg)
                raise FormFillerError(error_msg)

            try:
                logger.info("Filling form fields and OCR-detected fields...")
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

                if not success:
                    logger.warning("Some fields may not have been filled correctly.")
            except Exception as e:
                error_msg = f"Error during PDF filling: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                try:
                    # Cleanup temp file on error
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        logger.info(f"Cleaned up temporary file after error: {temp_output}")
                except Exception:
                    pass
                raise PDFWriteError(error_msg)

            try:
                self.finalize_pdf(temp_output, output_pdf)
                logger.info(f"✅ Finalized PDF saved to: {output_pdf}")

                # Cleanup temp file
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        logger.info(f"Cleaned up temporary file: {temp_output}")
                except Exception as cleanup_e:
                    logger.warning(f"Failed to clean up temporary file: {cleanup_e}")

                verification_result = self.verify_pdf_filled(output_pdf)
                if verification_result:
                    logger.info("PDF verification successful")
                else:
                    logger.warning("PDF verification indicates the form may not be properly filled")

                return verification_result

            except Exception as e:
                logger.error(f"Error during PDF finalization: {e}")
                logger.error(traceback.format_exc())
                logger.info("Trying alternative finalization method...")

                try:
                    shutil.copy2(temp_output, output_pdf)
                    logger.info(f"✅ Alternative save successful: {output_pdf}")

                    # Cleanup temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)

                    return self.verify_pdf_filled(output_pdf)

                except Exception as e2:
                    error_msg = f"Alternative save also failed: {e2}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    raise PDFWriteError(error_msg)

        except Exception as e:
            logger.error(f"Uncaught exception in match_and_fill_fields: {e}")
            logger.error(traceback.format_exc())
            raise

    def analyze_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                              ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze context around form fields to improve field understanding."""
        logger.info("Analyzing field context from OCR elements")
        field_context = []

        try:
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

                if self.debug_mode and nearby_text:
                    logger.debug(f"Field '{field_name}' has nearby text: {[t['text'] for t in nearby_text[:3]]}")

            logger.info(f"Context analysis complete for {len(field_context)} fields")
            return field_context

        except Exception as e:
            logger.error(f"Error during field context analysis: {e}")
            logger.error(traceback.format_exc())
            # Return empty context rather than failing
            return []

    def create_label_field_map(self, ocr_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create a map of potential form labels to nearby fields."""
        logger.info("Creating label-to-field mapping from OCR elements")
        label_map = {}

        try:
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

            logger.info(f"Created label map with {len(label_map)} potential form labels")
            return label_map

        except Exception as e:
            logger.error(f"Error creating label field map: {e}")
            logger.error(traceback.format_exc())
            # Return empty map rather than failing
            return {}

    def parse_ai_response(self, response_text: str) -> Dict[str, List]:
        """Parses AI response and extracts valid JSON matches for both form fields and OCR text."""
        logger.info("Parsing AI response...")

        try:
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
                        logger.warning(f"Skipping malformed field match: {match} | Error: {e}")

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
                        logger.warning(f"Skipping malformed OCR match: {match} | Error: {e}")

                logger.info(
                    f"Successfully parsed AI response: {len(result['field_matches'])} field matches, {len(result['ocr_matches'])} OCR matches")
                return result

            except json.JSONDecodeError as e:
                logger.error(f"AI returned invalid JSON: {e}")
                if self.debug_mode:
                    logger.debug(f"Failed text: {response_text[:500]}...")
                return {}

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.error(traceback.format_exc())
            return {}

    def fill_pdf_immediately(self, output_pdf: str, matches: List[FieldMatch],
                             pdf_fields: Dict[str, Dict[str, Any]]) -> bool:
        """Fills PDF form fields using PyMuPDF (fitz) with improved handling of readonly fields."""
        logger.info(f"Filling PDF form fields in: {output_pdf}")

        try:
            doc = fitz.open(output_pdf)
            filled_fields = []

            updates = []
            for match in matches:
                try:
                    if match.pdf_field and match.suggested_value is not None:
                        field_info = pdf_fields.get(match.pdf_field)

                        if not field_info:
                            logger.warning(f"Field '{match.pdf_field}' not found in PDF")
                            continue

                        if field_info["is_readonly"]:
                            logger.info(f"Skipping readonly field '{match.pdf_field}' - will handle via OCR")
                            continue

                        page_num = field_info["page_num"]
                        updates.append((page_num, match.pdf_field, match.suggested_value))
                except Exception as e:
                    logger.warning(f"Error processing match {match.pdf_field}: {e}")
                    continue

            logger.info(f"Preparing to update {len(updates)} fields")

            for page_num, field_name, value in updates:
                try:
                    page = doc[page_num]
                    for widget in page.widgets():
                        if widget.field_name == field_name:
                            logger.info(f"Filling: '{value}' to '{field_name}' (Page {page_num + 1})")
                            try:
                                widget.field_value = str(value)
                                widget.update()
                                filled_fields.append(field_name)
                            except Exception as e:
                                logger.warning(f"Error filling {field_name}: {e}")
                            break
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")

            try:
                # Remove garbage and incremental parameters to fix errors
                logger.info(f"Saving PDF with {len(filled_fields)} filled fields")
                doc.save(output_pdf, deflate=True, clean=True)
                doc.close()
                logger.info(f"✅ Successfully saved PDF with {len(filled_fields)} filled fields")
                return len(filled_fields) > 0

            except Exception as e:
                logger.error(f"Error saving PDF: {e}")
                logger.error(traceback.format_exc())

                try:
                    logger.info("Attempting alternative save method...")
                    temp_path = f"{output_pdf}.tmp"
                    doc.save(temp_path, deflate=True, clean=True)
                    doc.close()
                    shutil.move(temp_path, output_pdf)
                    logger.info(f"✅ Saved PDF using alternative method")
                    return len(filled_fields) > 0

                except Exception as e2:
                    logger.error(f"Alternative save also failed: {e2}")
                    logger.error(traceback.format_exc())
                    doc.close()
                    return False

        except Exception as e:
            logger.error(f"Uncaught exception in fill_pdf_immediately: {e}")
            logger.error(traceback.format_exc())
            raise PDFWriteError(f"Failed to fill PDF: {str(e)}")

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
                            f"✍️ Filling OCR field: '{match.suggested_value}' to near '{match.ocr_text}' (Page {match.page_num + 1})")

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
        form_filler = MultiAgentFormFiller()
        template_pdf = "D:\\demo\\Services\\PennsylvaniaLLC.pdf"
        json_path = "D:\\demo\\Services\\form_data.json"
        output_pdf = "D:\\demo\\Services\\filledform18.pdf"

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

        if success:
            print(f"✅ PDF successfully processed: {output_pdf}")
        else:
            print(f"❌ PDF processing failed. Please check the output file and logs.")

if __name__ == "__main__":
        asyncio.run(main())
