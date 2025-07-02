import asyncio
import json
import os
import re
import shutil
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import fitz
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, BooleanObject, ArrayObject
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator

from Common.constants import *

# Configure logging
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"tessofill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TessoFill")

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


class ProcessingError(Exception):
    """Base exception for TessoFill processing errors"""
    pass


class PDFExtractionError(ProcessingError):
    """Exception raised for errors in PDF field extraction"""
    pass


class OCRExtractionError(ProcessingError):
    """Exception raised for errors in OCR text extraction"""
    pass


class MatchingError(ProcessingError):
    """Exception raised for errors in field matching"""
    pass


class FillingError(ProcessingError):
    """Exception raised for errors in filling PDF fields"""
    pass


class MultiAgentFormFiller:
    def __init__(self):
        try:
            self.agent = Agent(
                model=GeminiModel("gemini-1.5-flash", api_key=API_KEYS["field_matcher"]),
                system_prompt="You are an expert at mapping PDF fields to JSON keys and filling them immediately."
            )
            logger.info("MultiAgentFormFiller initialized successfully")

            # Check if Tesseract is installed and accessible
            if os.name == 'nt':  # Windows
                pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                    logger.warning(
                        f"Tesseract not found at {pytesseract.pytesseract.tesseract_cmd}. OCR functionality may not work.")
        except Exception as e:
            logger.critical(f"Failed to initialize MultiAgentFormFiller: {str(e)}")
            raise

    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        """Extracts all fillable fields from a multi-page PDF with additional metadata and UUID focus."""
        logger.info(f"Extracting fillable fields from {pdf_path}")

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

                        # Extract UUID from field name if possible
                        uuid_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                                               field_name, re.IGNORECASE)
                        field_uuid = uuid_match.group(0) if uuid_match else field_name

                        field_type = widget.field_type
                        field_rect = widget.rect
                        field_flags = widget.field_flags

                        fields[field_uuid] = {
                            "original_name": field_name,
                            "page_num": page_num,
                            "type": field_type,
                            "rect": [field_rect.x0, field_rect.y0, field_rect.x1, field_rect.y1],
                            "flags": field_flags,
                            "is_readonly": bool(field_flags & 1),
                            "current_value": widget.field_value,
                            "uuid": field_uuid
                        }

                logger.info(f"Extracted {len(fields)} fields across {len(doc)} pages")
                for field_uuid, info in fields.items():
                    readonly_status = "READ-ONLY" if info["is_readonly"] else "EDITABLE"
                    logger.debug(f"Field: '{field_uuid}' (Page {info['page_num'] + 1}) [{readonly_status}]")

            doc.close()
            return fields
        except Exception as e:
            error_msg = f"Failed to extract PDF fields: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise PDFExtractionError(error_msg) from e

    async def extract_ocr_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using Tesseract OCR with position information and UUID detection."""
        logger.info(f"Extracting text using OCR from {pdf_path}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            doc = fitz.open(pdf_path)
            ocr_results = []
            uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

            for page_num in range(len(doc)):
                logger.info(f"Processing OCR for page {page_num + 1}/{len(doc)}")
                try:
                    pix = doc[page_num].get_pixmap(alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                    # Convert to grayscale for better OCR results
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    # Apply adaptive thresholding to improve text detection
                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2)

                    # Convert to PIL Image for Tesseract
                    pil_img = Image.fromarray(binary)

                    # Get OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT,
                                                         config='--oem 3 --psm 11')

                    # Process OCR results
                    seen_texts = set()
                    for i in range(len(ocr_data['text'])):
                        text = ocr_data['text'][i].strip()
                        conf = float(ocr_data['conf'][i]) / 100.0  # Convert confidence to 0-1 range

                        # Skip empty text or low confidence results
                        if not text or conf < 0.4:
                            continue

                        # Skip duplicates
                        if text in seen_texts:
                            continue

                        seen_texts.add(text)

                        # Get bounding box coordinates
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]

                        # Check if text contains a UUID
                        uuid_match = uuid_pattern.search(text)
                        is_uuid = bool(uuid_match)

                        # Add to results
                        ocr_results.append({
                            "page_num": page_num,
                            "text": text,
                            "raw_text": ocr_data['text'][i],
                            "confidence": conf,
                            "is_uuid": is_uuid,
                            "uuid": uuid_match.group(0) if is_uuid else None,
                            "position": {
                                "x1": float(x),
                                "y1": float(y),
                                "x2": float(x + w),
                                "y2": float(y + h)
                            }
                        })

                except Exception as page_error:
                    logger.error(f"Error processing OCR for page {page_num + 1}: {str(page_error)}")
                    logger.debug(traceback.format_exc())

            doc.close()

            # Filter and log UUID results
            uuid_results = [result for result in ocr_results if result['is_uuid']]
            logger.info(f"Extracted {len(uuid_results)} UUID elements out of {len(ocr_results)} total text elements")

            for uuid_result in uuid_results:
                logger.info(f"Found UUID: {uuid_result['uuid']} on page {uuid_result['page_num'] + 1}")

            return ocr_results
        except Exception as e:
            error_msg = f"Failed to extract OCR text: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise OCRExtractionError(error_msg) from e

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str,
                                    max_retries: int = 3) -> bool:
        """Matches fields using AI and fills them immediately across multiple pages, ensuring OCR text is mapped to UUIDs properly."""
        logger.info(f"Starting match_and_fill_fields process for {pdf_path}")

        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_pdf)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        backup_pdf = f"{pdf_path}.backup"
        try:
            shutil.copy2(pdf_path, backup_pdf)
            logger.info(f"Created backup of original PDF: {backup_pdf}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")

        try:
            pdf_fields = await self.extract_pdf_fields(pdf_path)
        except PDFExtractionError as e:
            logger.error(f"PDF field extraction failed: {str(e)}")
            return False

        try:
            ocr_text_elements = await self.extract_ocr_text(pdf_path)
        except OCRExtractionError as e:
            logger.warning(f"OCR extraction failed: {str(e)}. Continuing with limited functionality.")
            ocr_text_elements = []

        try:
            flat_json = self.flatten_json(json_data)
            field_context = self.analyze_field_context(pdf_fields, ocr_text_elements)

            # Log available JSON fields for debugging
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

            matches, ocr_matches = [], []
            for attempt in range(max_retries):
                try:
                    logger.info(f"AI matching attempt {attempt + 1}/{max_retries}")
                    response = await self.agent.run(prompt)
                    result = self.parse_ai_response(response.data)

                    if result:
                        matches = result.get("field_matches", [])
                        ocr_matches = result.get("ocr_matches", [])
                        if matches or ocr_matches:
                            logger.info(
                                f"Successfully matched {len(matches)} fields and {len(ocr_matches)} OCR elements")
                            break
                except Exception as e:
                    logger.error(f"Error during AI matching attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise MatchingError(f"All AI matching attempts failed: {str(e)}") from e
                    else:
                        logger.info(f"Retrying AI matching...")

            if not matches and not ocr_matches:
                logger.warning("No valid field matches were found after all attempts")
                return False

            temp_output = f"{output_pdf}.temp"
            shutil.copy2(pdf_path, temp_output)
            logger.info(f"Created temporary working file: {temp_output}")

            logger.info("Filling form fields and OCR-detected fields together with UUID-based matching")
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
                logger.warning("Some fields may not have been filled correctly")

            try:
                self.finalize_pdf(temp_output, output_pdf)
                logger.info(f"Finalized PDF saved to: {output_pdf}")
                verification_result = self.verify_pdf_filled(output_pdf)
                if verification_result:
                    logger.info("PDF verification successful")
                else:
                    logger.warning("PDF verification failed - output may be incomplete")
                return verification_result
            except Exception as e:
                logger.error(f"Error during finalization: {str(e)}")
                logger.debug(traceback.format_exc())
                logger.info("Trying alternative finalization method")

                try:
                    shutil.copy2(temp_output, output_pdf)
                    logger.info(f"Alternative save successful: {output_pdf}")
                    verification_result = self.verify_pdf_filled(output_pdf)
                    if verification_result:
                        logger.info("PDF verification successful")
                    else:
                        logger.warning("PDF verification failed - output may be incomplete")
                    return verification_result
                except Exception as e2:
                    logger.error(f"Alternative save also failed: {str(e2)}")
                    logger.debug(traceback.format_exc())
                    return False

        except MatchingError as e:
            logger.error(f"Matching process failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in match_and_fill_fields: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
        finally:
            # Cleanup temp files
            try:
                temp_files = [f"{output_pdf}.temp", f"{output_pdf}.temp.tmp"]
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {str(e)}")

    def analyze_field_context(self, pdf_fields: Dict[str, Dict[str, Any]],
                              ocr_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze context around form fields to improve field understanding."""
        logger.info("Analyzing field context from OCR data")
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

            logger.debug(f"Analyzed context for {len(field_context)} fields")
            return field_context
        except Exception as e:
            logger.error(f"Error analyzing field context: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return empty context rather than failing
            return []

    def create_label_field_map(self, ocr_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create a map of potential form labels to nearby fields."""
        logger.info("Creating label-to-field mapping from OCR data")
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

            logger.debug(f"Created label map with {len(label_map)} entries")
            return label_map
        except Exception as e:
            logger.error(f"Error creating label field map: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}

    def parse_ai_response(self, response_text: str) -> Dict[str, List]:
        """Parses AI response and extracts valid JSON matches for both form fields and OCR text."""
        logger.info("Parsing AI response")
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
        logger.debug(f"Extracted JSON text from response: {response_text[:100]}...")

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
                    logger.warning(f"Skipping malformed field match: {match} | Error: {str(e)}")

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
                    logger.warning(f"Skipping malformed OCR match: {match} | Error: {str(e)}")

            logger.info(
                f"Successfully parsed {len(result['field_matches'])} field matches and {len(result['ocr_matches'])} OCR matches")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"AI returned invalid JSON: {str(e)}")
            logger.debug(f"Failed text: {response_text[:500]}...")
            return {}
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}

    def fill_pdf_immediately(self, output_pdf: str, matches: List[FieldMatch],
                             pdf_fields: Dict[str, Dict[str, Any]]) -> bool:
        """Fills PDF form fields using PyMuPDF (fitz) with improved handling of readonly fields."""
        logger.info(f"Filling PDF fields in {output_pdf}")
        if not os.path.exists(output_pdf):
            logger.error(f"PDF file not found: {output_pdf}")
            return False

        try:
            doc = fitz.open(output_pdf)
            filled_fields = []

            updates = []
            for match in matches:
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

            for page_num, field_name, value in updates:
                page = doc[page_num]
                field_filled = False
                for widget in page.widgets():
                    if widget.field_name == field_name:
                        logger.info(f"Filling: '{value}' → '{field_name}' (Page {page_num + 1})")
                        try:
                            widget.field_value = str(value)
                            widget.update()
                            filled_fields.append(field_name)
                            field_filled = True
                        except Exception as e:
                            logger.error(f"Error filling {field_name}: {str(e)}")
                            logger.debug(traceback.format_exc())
                        break
                if not field_filled:
                    logger.warning(f"Could not find widget for field {field_name} on page {page_num + 1}")

            try:
                # Remove garbage and incremental parameters to fix the error
                doc.save(output_pdf, deflate=True, clean=True)
                logger.info(f"Saved PDF with {len(filled_fields)} filled fields")
                doc.close()
                return len(filled_fields) > 0
            except Exception as e:
                logger.error(f"Error saving PDF: {str(e)}")
                logger.debug(traceback.format_exc())

                try:
                    temp_path = f"{output_pdf}.tmp"
                    doc.save(temp_path, deflate=True, clean=True)
                    doc.close()
                    shutil.move(temp_path, output_pdf)
                    logger.info(f"Saved PDF using alternative method")
                    return len(filled_fields) > 0
                except Exception as e2:
                    logger.error(f"Alternative save also failed: {str(e2)}")
                    logger.debug(traceback.format_exc())
                    doc.close()
                    return False
        except Exception as e:
            logger.error(f"Unexpected error in fill_pdf_immediately: {str(e)}")
            logger.debug(traceback.format_exc())
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
    form_filler = MultiAgentFormFiller()
    template_pdf = "D:\\demo\\Services\\PennsylvaniaLLC.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
    output_pdf = "D:\\demo\\Services\\fill_smart4.pdf"

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    success = await form_filler.match_and_fill_fields(template_pdf, json_data, output_pdf)

    if success:
        print(f"✅ PDF successfully processed: {output_pdf}")
    else:
        print(f"❌ PDF processing failed. Please check the output file and logs.")


if __name__ == "__main__":
    asyncio.run(main())
