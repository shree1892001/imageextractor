import asyncio
import json
import os
import re
import numpy as np
import cv2
import fitz
import time
from typing import Dict, Any, List, Tuple, Optional
import traceback
import shutil
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, field_validator, ConfigDict, ValidationError
from Common.constants import *

# Add timeout constants
API_TIMEOUT = 30  # seconds
OCR_TIMEOUT = 60  # seconds

API_KEYS = {
    "field_matcher": "YOUR_API_KEY",  # Replace with your actual API key
}


class AIResponseValidationError(Exception):
    """Custom exception for AI response validation failures."""
    pass


class AdvancedOCRProcessor:
    def __init__(self, languages=['en'], use_gpu=True):
        """
        Initialize advanced OCR processor with multi-language support and GPU acceleration

        Args:
            languages (List[str]): List of language codes to support
            use_gpu (bool): Enable GPU acceleration for OCR
        """
        print("Initializing OCR processor...")
        self.ocr_readers = {}
        for lang in languages:
            try:
                self.ocr_readers[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    use_gpu=use_gpu,
                    show_log=False
                )
                print(f"Initialized OCR reader for language: {lang}")
            except Exception as e:
                print(f"Failed to initialize OCR reader for language {lang}: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing for improved OCR accuracy

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        print("Preprocessing image...")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        print("Image preprocessing complete")

        return denoised

    def extract_text_multilingual(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text using multiple language OCR with advanced preprocessing

        Args:
            image (np.ndarray): Input image

        Returns:
            List[Dict[str, Any]]: Extracted text with details
        """
        print("Starting OCR text extraction...")
        start_time = time.time()

        preprocessed_image = self.preprocess_image(image)

        all_results = []
        for lang, ocr_reader in self.ocr_readers.items():
            try:
                print(f"Running OCR for language: {lang}")
                # Add timeout check
                if time.time() - start_time > OCR_TIMEOUT:
                    print(f"OCR timeout after {OCR_TIMEOUT} seconds. Returning partial results.")
                    break

                results = ocr_reader.ocr(preprocessed_image, cls=True)
                # Convert generator to list if needed
                if hasattr(results, '__iter__') and not hasattr(results, '__len__'):
                    results = list(results)

                print(f"OCR completed for language {lang}")
                if results:
                    for result in results:
                        # Convert inner result to list if it's a generator
                        if hasattr(result, '__iter__') and not hasattr(result, '__len__'):
                            result = list(result)

                        for text_info in result:
                            all_results.append({
                                'text': text_info[1][0],
                                'confidence': text_info[1][1],
                                'bbox': text_info[0],
                                'language': lang
                            })
            except Exception as e:
                print(f"OCR error for language {lang}: {e}")

        # Sort results by confidence
        results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
        print(f"Extracted {len(results)} text elements in {time.time() - start_time:.2f} seconds")
        return results


class PlaceholderDetector:
    """Enhanced class to detect and parse placeholders in parentheses"""

    @staticmethod
    def detect_form_fields(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Detect form fields in PDF that are likely input fields

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[Dict[str, Any]]: List of detected form fields with coordinates
        """
        print(f"Detecting form fields in {pdf_path}...")
        form_fields = []

        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            print(f"Analyzing page {page_idx + 1}/{len(doc)} for form fields")

            # Method 1: Look for underscores and boxes that might indicate input fields
            text = page.get_text("dict")
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content = span["text"]
                        # Check for form field indicators
                        if "_" * 3 in text_content or "□" in text_content or "■" in text_content:
                            form_fields.append({
                                'type': 'input_field',
                                'page': page_idx,
                                'bbox': span["bbox"],
                                'text': text_content
                            })

            # Method 2: Look for rectangles that might be form field boxes
            paths = page.get_drawings()
            for path in paths:
                for item in path["items"]:
                    if item[0] == "re":  # Rectangle
                        rect = item[1]  # Rectangle coordinates
                        # Filter for rectangles that are likely to be form fields
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        if 20 < width < 300 and 5 < height < 40:
                            form_fields.append({
                                'type': 'input_box',
                                'page': page_idx,
                                'bbox': rect
                            })

        print(f"Detected {len(form_fields)} potential form fields")
        return form_fields

    @staticmethod
    def extract_placeholders(ocr_results: List[Dict[str, Any]], form_fields: List[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """
        Extract placeholders in parentheses from OCR results, prioritizing those within form fields

        Args:
            ocr_results (List[Dict[str, Any]]): OCR text elements
            form_fields (List[Dict[str, Any]], optional): Detected form fields

        Returns:
            List[Dict[str, Any]]: Extracted placeholders with position info
        """
        print("Extracting placeholders from OCR results...")
        placeholders = []

        # Regular expression to match text in parentheses
        placeholder_pattern = r'\((.*?)\)'

        # First, create a mapping of pages to form field areas
        form_field_areas = {}
        if form_fields:
            for field in form_fields:
                page = field.get('page', 0)
                if page not in form_field_areas:
                    form_field_areas[page] = []
                form_field_areas[page].append(field)

        # Process OCR results
        for result in ocr_results:
            text = result.get('text', '')
            matches = re.findall(placeholder_pattern, text)

            if matches and 'bbox' in result:
                page = result.get('page', 0)

                # Check if this OCR result is within or near a form field
                is_in_form_field = False
                nearest_field = None

                if page in form_field_areas:
                    for field in form_field_areas[page]:
                        field_bbox = field.get('bbox')
                        ocr_bbox = result.get('bbox')

                        # Check for overlap or proximity
                        if PlaceholderDetector._bbox_overlap_or_near(ocr_bbox, field_bbox, threshold=20):
                            is_in_form_field = True
                            nearest_field = field
                            break

                # Process placeholders, prioritizing those in form fields
                for match in matches:
                    # Create a placeholder entry with necessary info
                    placeholder = {
                        'placeholder_text': match.strip(),
                        'original_text': text,
                        'page': page,
                        'bbox': result['bbox'],
                        'confidence': result.get('confidence', 0),
                        'is_in_form_field': is_in_form_field
                    }

                    # Add form field info if available
                    if nearest_field:
                        placeholder['field_type'] = nearest_field.get('type')
                        placeholder['field_bbox'] = nearest_field.get('bbox')

                    placeholders.append(placeholder)
                    print(f"Found placeholder: '{match.strip()}' on page {page} - In form field: {is_in_form_field}")

        # Sort placeholders - prioritize those in form fields
        placeholders.sort(key=lambda x: (not x.get('is_in_form_field', False), -x.get('confidence', 0)))

        print(f"Extracted {len(placeholders)} placeholders in total")
        return placeholders

    @staticmethod
    @staticmethod
    def _bbox_overlap_or_near(bbox1, bbox2, threshold=20):
        """
        Check if two bounding boxes overlap or are near each other

        Args:
            bbox1: First bounding box [x0, y0, x1, y1]
            bbox2: Second bounding box [x0, y0, x1, y1]
            threshold: Distance threshold for "nearness"

        Returns:
            bool: True if bounding boxes overlap or are near each other
        """
        # Ensure both bboxes are proper lists/tuples of 4 values
        if not (isinstance(bbox1, (list, tuple)) and len(bbox1) == 4 and
                isinstance(bbox2, (list, tuple)) and len(bbox2) == 4):
            print(f"Warning: Invalid bbox format. bbox1: {bbox1}, bbox2: {bbox2}")
            return False

        # Unpack coordinates
        try:
            x0_1, y0_1, x1_1, y1_1 = map(float, bbox1)
            x0_2, y0_2, x1_2, y1_2 = map(float, bbox2)
        except (ValueError, TypeError) as e:
            print(f"Error converting bbox values to float: {e}")
            print(f"bbox1: {bbox1}, bbox2: {bbox2}")
            return False

        # Check for overlap
        if (x0_1 < x1_2 and x1_1 > x0_2 and y0_1 < y1_2 and y1_1 > y0_2):
            return True

        # Check for nearness
        # Find the closest points between the two boxes
        x_dist = max(0, max(x0_1, x0_2) - min(x1_1, x1_2))
        y_dist = max(0, max(y0_1, y0_2) - min(y1_1, y1_2))

        return (x_dist < threshold) and (y_dist < threshold)

class PlaceholderFieldMatchingAgent:
    """Specialized agent for matching placeholders to JSON fields"""

    def __init__(self, model_name="gemini-1.5-flash"):
        """Initialize the placeholder matching agent"""
        print(f"Initializing Placeholder Matching Agent with model: {model_name}")
        try:
            self.agent = Agent(
                model=GeminiModel(model_name, api_key=API_KEYS.get("field_matcher")),
                system_prompt="""
                You are an AI specialized in matching PDF form placeholders to JSON data fields.

                Your task is to examine placeholders found in a PDF form (text in parentheses like "(name)" or 
                "(address)") and determine which JSON field they correspond to.

                CRITICAL OUTPUT REQUIREMENTS:
                ALWAYS return a VALID JSON with this EXACT structure:
                {
                    "placeholder_matches": [
                        {
                            "placeholder": "name",  // The placeholder text (without parentheses)
                            "json_field": "person.fullName",  // The corresponding JSON field path
                            "suggested_value": "John Smith",  // The value from the JSON to insert
                            "confidence": 0.9,  // Your confidence in this match (0.0-1.0)
                            "reasoning": "The placeholder requests a name which matches the person.fullName field"
                        }
                    ]
                }

                DO NOT include code blocks, comments, or additional text outside the JSON structure.
                """
            )
            print("Placeholder Matching Agent initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Placeholder Matching Agent: {e}")
            raise

    async def match_placeholders(
            self,
            json_data: Dict[str, Any],
            placeholders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match placeholders to JSON fields

        Args:
            json_data: The JSON data with values to insert
            placeholders: List of extracted placeholders with position info

        Returns:
            List of placeholder matches with suggested values
        """
        print(f"Matching {len(placeholders)} placeholders to JSON fields...")
        start_time = time.time()

        # Prioritize placeholders in form fields
        form_field_placeholders = [p for p in placeholders if p.get('is_in_form_field', False)]
        other_placeholders = [p for p in placeholders if not p.get('is_in_form_field', False)]

        prioritized_placeholders = form_field_placeholders + other_placeholders

        # Use at most 50 placeholders to avoid overwhelming the model
        max_placeholders = 50
        truncated_placeholders = prioritized_placeholders[:max_placeholders]

        print(f"Selected {len(truncated_placeholders)} priority placeholders for matching")

        # Prepare the placeholder matching prompt
        prompt = json.dumps({
            "json_data": json_data,
            "placeholders": [
                {
                    "placeholder_text": p["placeholder_text"],
                    "page": p.get("page", 0),
                    "is_in_form_field": p.get("is_in_form_field", False),
                    "field_type": p.get("field_type", "unknown"),
                    "context": p.get("original_text", "")
                } for p in truncated_placeholders
            ],
            "instructions": "Match each placeholder to the most appropriate JSON field, prioritizing placeholders found in form fields. Consider both semantic matching and context from surrounding text. For example, '(name)' might match to 'person.name' or 'company.name' depending on context."
        }, indent=2)

        try:
            print("Sending request to AI model for placeholder matching...")
            response_future = self.agent.run(prompt)
            response = await asyncio.wait_for(response_future, timeout=API_TIMEOUT)
            print(f"Received AI placeholder matching response in {time.time() - start_time:.2f} seconds")
            return self._parse_and_validate_placeholder_matches(response.data)
        except asyncio.TimeoutError:
            print(f"AI request timed out after {API_TIMEOUT} seconds")
            return []
        except Exception as e:
            print(f"Placeholder matching error: {e}")
            return []

    def _parse_and_validate_placeholder_matches(self, raw_response: str) -> List[Dict[str, Any]]:
        """Parse and validate the placeholder matches from AI response"""
        print("Parsing placeholder matching response...")

        try:
            # Clean response
            clean_response = re.sub(r'```(json)?', '', raw_response).strip()

            # Parse JSON
            try:
                parsed_matches = json.loads(clean_response)
            except json.JSONDecodeError as json_err:
                print(f"JSON Decoding Error: {json_err}")
                print(f"First 200 chars of problematic response: {clean_response[:200]}")
                return []

            # Extract and validate matches
            matches = []
            match_count = len(parsed_matches.get('placeholder_matches', []))
            print(f"Found {match_count} raw placeholder matches to validate")

            for match in parsed_matches.get('placeholder_matches', []):
                if not all(key in match for key in ['placeholder', 'json_field', 'suggested_value']):
                    print(f"Incomplete placeholder match: {match}")
                    continue

                matches.append({
                    'placeholder': str(match['placeholder']),
                    'json_field': str(match['json_field']),
                    'suggested_value': str(match['suggested_value']),
                    'confidence': float(match.get('confidence', 0.7)),
                    'reasoning': str(match.get('reasoning', 'No reasoning provided'))
                })

            print(f"Parsed {len(matches)} valid placeholder matches")
            return matches

        except Exception as e:
            print(f"Placeholder match parsing error: {e}")
            return []


class FormFieldAnalyzer:
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initialize AI agent for form field analysis and placement

        Args:
            model_name (str): AI model to use
        """
        print(f"Initializing Form Field Analyzer with model: {model_name}")
        try:
            self.agent = Agent(
                model=GeminiModel(model_name, api_key=API_KEYS.get("field_matcher")),
                system_prompt="""
                You are an AI specialized in analyzing PDF form fields and determining the exact positions 
                where values should be placed, especially for placeholder replacement.

                When analyzing placeholders in form fields, you need to identify:
                1. The page number where the placeholder is located
                2. The precise coordinates where the replacement text should be inserted
                3. Any special formatting requirements
                4. The available space for the text

                CRITICAL OUTPUT REQUIREMENTS:
                ALWAYS return a VALID JSON with this EXACT structure:
                {
                    "placeholder_positions": [
                        {
                            "placeholder_text": "name",  // The placeholder text without parentheses
                            "page": 0,  // 0-indexed page number
                            "coordinates": {
                                "x": 150.5,  // x-coordinate for text insertion
                                "y": 225.3   // y-coordinate for text insertion
                            },
                            "max_width": 200,  // maximum width allowed for inserted text
                            "formatting": {
                                "alignment": "left",  // left, center, right
                                "font_size": 10
                            }
                        }
                    ]
                }

                DO NOT include code blocks, comments, or additional text outside the JSON structure.
                """
            )
            print("Form Field Analyzer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Form Field Analyzer: {e}")
            raise

    async def analyze_placeholder_positions(
            self,
            placeholders: List[Dict[str, Any]],
            pdf_width: int,
            pdf_height: int,
            placeholder_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze placeholder positions to determine optimal text placement

        Args:
            placeholders: List of detected placeholders with position info
            pdf_width: Width of the PDF in points
            pdf_height: Height of the PDF in points
            placeholder_matches: List of placeholder matches to be filled

        Returns:
            List of placeholder position mappings with exact coordinates
        """
        print("Starting placeholder position analysis...")
        start_time = time.time()

        # Filter placeholders to only those that were matched
        matched_placeholder_texts = [m['placeholder'] for m in placeholder_matches]
        relevant_placeholders = [
            p for p in placeholders
            if p['placeholder_text'] in matched_placeholder_texts
        ]

        print(f"Analyzing positions for {len(relevant_placeholders)} matched placeholders")

        # Limit to 30 placeholders max to avoid overwhelming the model
        if len(relevant_placeholders) > 30:
            # Prioritize placeholders in form fields
            form_field_placeholders = [p for p in relevant_placeholders if p.get('is_in_form_field', False)]
            other_placeholders = [p for p in relevant_placeholders if not p.get('is_in_form_field', False)]

            relevant_placeholders = form_field_placeholders + other_placeholders
            relevant_placeholders = relevant_placeholders[:30]
            print(f"Limited analysis to 30 prioritized placeholders")

        # Prepare analysis prompt
        prompt_data = {
            "pdf_dimensions": {
                "width": pdf_width,
                "height": pdf_height
            },
            "placeholders": [],
            "instructions": "Analyze each placeholder's position and determine the exact coordinates where replacement text should be placed. For placeholders in form fields, ensure text is properly positioned within the field boundaries."
        }

        # Add detailed placeholder information
        for placeholder in relevant_placeholders:
            placeholder_info = {
                "placeholder_text": placeholder["placeholder_text"],
                "page": placeholder.get("page", 0),
                "bbox": placeholder.get("bbox"),
                "is_in_form_field": placeholder.get("is_in_form_field", False)
            }

            # Add form field info if available
            if placeholder.get("field_bbox"):
                placeholder_info["field_bbox"] = placeholder.get("field_bbox")
                placeholder_info["field_type"] = placeholder.get("field_type", "unknown")

            # Add matched value for context on sizing
            match = next((m for m in placeholder_matches if m['placeholder'] == placeholder["placeholder_text"]), None)
            if match:
                placeholder_info["replacement_value"] = match.get("suggested_value", "")

            prompt_data["placeholders"].append(placeholder_info)

        prompt = json.dumps(prompt_data, indent=2)

        try:
            print("Sending request to AI model for placeholder position analysis...")
            response_future = self.agent.run(prompt)
            response = await asyncio.wait_for(response_future, timeout=API_TIMEOUT)
            print(f"Received AI position analysis response in {time.time() - start_time:.2f} seconds")
            return self._parse_and_validate_positions(response.data)
        except asyncio.TimeoutError:
            print(f"AI request timed out after {API_TIMEOUT} seconds")
            return []
        except Exception as e:
            print(f"Placeholder position analysis error: {e}")
            return []

    def _parse_and_validate_positions(self, raw_response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate the AI response for placeholder positions

        Args:
            raw_response: Raw AI response

        Returns:
            List of validated placeholder positions
        """
        print("Parsing placeholder position analysis...")

        try:
            # Clean response
            clean_response = re.sub(r'```(json)?', '', raw_response).strip()
            print(f"Cleaned response length: {len(clean_response)} characters")
            print(f"Response preview: {clean_response[:100]}...")

            # Parse JSON
            try:
                parsed_positions = json.loads(clean_response)
            except json.JSONDecodeError as json_err:
                print(f"JSON Decoding Error: {json_err}")
                print(f"First 200 chars of problematic response: {clean_response[:200]}")
                return []

            # Extract and validate positions
            placeholder_positions = []
            position_count = len(parsed_positions.get('placeholder_positions', []))
            print(f"Found {position_count} raw positions to validate")

            for position in parsed_positions.get('placeholder_positions', []):
                if not all(key in position for key in ['placeholder_text', 'page', 'coordinates']):
                    print(f"Incomplete position data: {position}")
                    continue

                # Normalize formatting if missing
                if 'formatting' not in position:
                    position['formatting'] = {
                        'alignment': 'left',
                        'font_size': 10
                    }

                # Normalize max_width if missing
                if 'max_width' not in position:
                    position['max_width'] = 200

                placeholder_positions.append(position)

            print(f"Parsed {len(placeholder_positions)} valid placeholder positions")
            return placeholder_positions

        except Exception as e:
            print(f"Position parsing error: {e}")
            return []


class PDFSmartFiller:
    def __init__(self):
        print("Initializing PDF Smart Filler...")
        self.ocr_processor = AdvancedOCRProcessor()
        self.placeholder_detector = PlaceholderDetector()
        self.placeholder_matcher = PlaceholderFieldMatchingAgent()
        self.field_analyzer = FormFieldAnalyzer()
        print("PDF Smart Filler initialized")

    async def process_pdf(
            self,
            input_pdf_path: str,
            json_data: Dict[str, Any],
            output_pdf_path: str
    ):
        """
        Comprehensive PDF processing focusing on placeholders in form fields

        Args:
            input_pdf_path (str): Input PDF file path
            json_data (Dict[str, Any]): Data to fill in PDF
            output_pdf_path (str): Output filled PDF path
        """
        print(f"Starting PDF processing for: {input_pdf_path}")
        overall_start_time = time.time()

        try:
            # Verify input file exists
            if not os.path.exists(input_pdf_path):
                print(f"Error: Input PDF not found at {input_pdf_path}")
                return

            # Verify output directory exists
            output_dir = os.path.dirname(output_pdf_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")

            # First, detect form fields in the PDF
            print("Detecting form fields in PDF...")
            form_fields = self.placeholder_detector.detect_form_fields(input_pdf_path)
            print(f"Detected {len(form_fields)} potential form fields")

            # Convert PDF to images for OCR
            print(f"Converting PDF to images: {input_pdf_path}")
            pdf_conversion_start = time.time()
            pdf_images = convert_from_path(input_pdf_path)
            # Convert to list if it's a generator
            if hasattr(pdf_images, '__iter__') and not hasattr(pdf_images, '__len__'):
                pdf_images = list(pdf_images)
            pdf_conversion_time = time.time() - pdf_conversion_start
            print(f"PDF conversion completed in {pdf_conversion_time:.2f} seconds. Found {len(pdf_images)} pages.")

            # Extract text from images with progress reporting
            print("Starting OCR text extraction...")
            ocr_start_time = time.time()

            all_ocr_results = []
            for page_num, image in enumerate(pdf_images):
                page_start_time = time.time()
                print(f"Processing page {page_num + 1}/{len(pdf_images)}")

                img_array = np.array(image)
                page_results = self.ocr_processor.extract_text_multilingual(img_array)

                # Add page number to results
                for result in page_results:
                    result['page'] = page_num

                all_ocr_results.extend(page_results)

                page_time = time.time() - page_start_time
                print(
                    f"Page {page_num + 1} processed in {page_time:.2f} seconds. Found {len(page_results)} text elements.")

            ocr_time = time.time() - ocr_start_time
            print(f"OCR completed in {ocr_time:.2f} seconds. Found {len(all_ocr_results)} total text elements.")

            # Extract placeholders from OCR results, prioritizing those in form fields
            print("Extracting placeholders from OCR results...")
            placeholder_start_time = time.time()
            placeholders = self.placeholder_detector.extract_placeholders(
                all_ocr_results,
                form_fields
            )
            placeholder_time = time.time() - placeholder_start_time
            print(
                f"Placeholder extraction completed in {placeholder_time:.2f} seconds. Found {len(placeholders)} placeholders."
            )

            # Count placeholders in form fields vs. outside
            form_field_placeholders = [p for p in placeholders if p.get('is_in_form_field', False)]
            print(
                f"Found {len(form_field_placeholders)} placeholders in form fields and {len(placeholders) - len(form_field_placeholders)} outside form fields.")

            # Match placeholders to JSON fields
            print("Matching placeholders to JSON fields...")
            if placeholders:
                placeholder_match_start_time = time.time()
                placeholder_matches = await self.placeholder_matcher.match_placeholders(
                    json_data,
                    placeholders
                )
                placeholder_match_time = time.time() - placeholder_match_start_time
                print(
                    f"Placeholder matching completed in {placeholder_match_time:.2f} seconds. Found {len(placeholder_matches)} matches.")
            else:
                placeholder_matches = []
                print("No placeholders found to match.")

            # Analyze placeholder positions
            if placeholder_matches:
                print("Analyzing placeholder positions...")

                # Get PDF dimensions
                doc = fitz.open(input_pdf_path)
                first_page = doc[0]
                pdf_width, pdf_height = first_page.rect.width, first_page.rect.height
                print(f"PDF dimensions: {pdf_width}x{pdf_height}")

                analysis_start_time = time.time()
                placeholder_positions = await self.field_analyzer.analyze_placeholder_positions(
                    placeholders,
                    pdf_width,
                    pdf_height,
                    placeholder_matches
                )
                doc.close()

                analysis_time = time.time() - analysis_start_time
                print(
                    f"Placeholder position analysis completed in {analysis_time:.2f} seconds. Found {len(placeholder_positions)} positions.")

                # Fill PDF with placeholder values
                if placeholder_positions:
                    filling_start_time = time.time()
                    await self._fill_pdf_with_placeholder_values(
                        input_pdf_path,
                        output_pdf_path,
                        placeholder_matches,
                        placeholder_positions
                    )
                    filling_time = time.time() - filling_start_time
                    print(f"PDF filling completed in {filling_time:.2f} seconds.")
                else:
                    print("No valid placeholder positions found. Skipping PDF filling.")
                    return
            else:
                print("No placeholder matches found. Skipping PDF filling.")
                return

            total_time = time.time() - overall_start_time
            print(f"Overall processing completed in {total_time:.2f} seconds.")
            print(f"Output PDF saved to: {output_pdf_path}")

        except Exception as e:
            print(f"Error during PDF processing: {e}")
            print(f"Processing failed for {input_pdf_path}")
            traceback.print_exc()

    async def _fill_pdf_with_ai_positions(
                self,
                input_pdf_path: str,
                output_pdf_path: str,
                field_matches: List[Dict[str, Any]],
                field_positions: List[Dict[str, Any]]
        ):
            """
            Replace placeholders in PDF with values from field_matches using positions from field_positions.
            """
            print(f"Starting PDF placeholder replacement with {len(field_matches)} values...")

            try:

                input_pdf_path = os.path.abspath(input_pdf_path)
                output_pdf_path = os.path.abspath(output_pdf_path)

                # Create a temporary path in a different location (not derived from input_path)
                temp_dir = os.path.dirname(output_pdf_path)
                temp_filename = f"temp_{os.path.basename(input_pdf_path)}"
                temp_path = os.path.join(temp_dir, temp_filename)
                if temp_path == input_pdf_path or temp_path == output_pdf_path:
                    temp_path = os.path.join(temp_dir, f"temp2_{os.path.basename(input_pdf_path)}")

                    # Copy the input PDF to the temp file
                shutil.copy(input_pdf_path, temp_path)

                doc = fitz.open(temp_path)
                print(f"Opened PDF with {len(doc)} pages")

                replacement_map = {}
                for match in field_matches:
                    pdf_field = match['pdf_field']
                    value = match['suggested_value']
                    replacement_map[pdf_field] = value

                    if not (pdf_field.startswith('(') and pdf_field.endswith(')')):
                        placeholder = f"({pdf_field})"
                        replacement_map[placeholder] = value
                        print(f"Created mapping for: {placeholder} -> {value}")
                    else:
                        print(f"Mapping placeholder: {pdf_field} -> {value}")

                replacement_count = 0

                all_replacements = []

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    print(f"First pass - analyzing page {page_num + 1}")

                    text_instances = page.get_text("dict")

                    page_replacements = []

                    for block in text_instances.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                original_text = span["text"]
                                rect = fitz.Rect(span["bbox"])

                                if rect.width < 2 or rect.height < 2:
                                    continue

                                placeholder_found = False
                                replacement_text = original_text

                                for pattern, replacement in replacement_map.items():
                                    if pattern in original_text:
                                        replacement_text = replacement_text.replace(pattern, replacement)
                                        placeholder_found = True

                                if not placeholder_found:
                                    placeholder_pattern = r'\((.*?)\)'
                                    matches = re.findall(placeholder_pattern, original_text)
                                    for match in matches:
                                        placeholder = f"({match})"
                                        if placeholder in replacement_map:
                                            replacement_text = replacement_text.replace(
                                                placeholder, replacement_map[placeholder]
                                            )
                                            placeholder_found = True

                                if placeholder_found and replacement_text != original_text:
                                    padding = 8
                                    redact_rect = fitz.Rect(
                                        rect.x0 - padding,
                                        rect.y0 - padding,
                                        rect.x1 + padding,
                                        rect.y1 + padding
                                    )

                                    page_replacements.append({
                                        'rect': redact_rect,
                                        'new_text': replacement_text,
                                        'font_size': span.get("size", 11),
                                        'text_origin': (rect.x0, rect.y0),
                                        'original_text': original_text
                                    })
                                    print(f"Found text to replace: '{original_text}' -> '{replacement_text}'")

                    for position in field_positions:
                        field_name = position.get('field_name')
                        pos_page_num = position.get('page', 0)

                        if pos_page_num == page_num:
                            matching_field = next((m for m in field_matches if m['pdf_field'] == field_name), None)

                            if matching_field:
                                value = matching_field['suggested_value']
                                coords = position.get('coordinates', {})
                                x = coords.get('x', 0)
                                y = coords.get('y', 0)
                                formatting = position.get('formatting', {})
                                font_size = formatting.get('font_size', 11)
                                alignment = formatting.get('alignment', 'left')
                                max_width = position.get('max_width', 200)

                                position_rect = fitz.Rect(
                                    x - 5,
                                    y - font_size - 5,
                                    x + max_width + 5,
                                    y + 5
                                )

                                page_replacements.append({
                                    'rect': position_rect,
                                    'new_text': value,
                                    'font_size': font_size,
                                    'text_origin': (x, y),
                                    'alignment': alignment,
                                    'max_width': max_width,
                                    'is_position': True
                                })
                                print(f"Found position for field '{field_name}' at ({x}, {y})")

                    if page_replacements:
                        all_replacements.append({
                            'page_num': page_num,
                            'replacements': page_replacements
                        })

                for page_data in all_replacements:
                    page_num = page_data['page_num']
                    page = doc[page_num]
                    print(f"Second pass - processing page {page_num + 1}")

                    new_page = doc.new_page(page_num + 1, width=page.rect.width, height=page.rect.height)
                    new_page.show_pdf_page(new_page.rect, doc, page_num)

                    doc.delete_page(page_num)

                    page = doc[page_num]

                    replacements = page_data['replacements']

                    grouped_rects = []
                    for repl in replacements:
                        rect = repl['rect']
                        added_to_group = False

                        for group in grouped_rects:

                            for existing_rect in group:
                                if (rect.x0 < existing_rect.x1 and rect.x1 > existing_rect.x0 and
                                        rect.y0 < existing_rect.y1 and rect.y1 > existing_rect.y0):
                                    group.append(rect)
                                    added_to_group = True
                                    break
                            if added_to_group:
                                break

                        if not added_to_group:
                            grouped_rects.append([rect])

                    merged_rects = []
                    for group in grouped_rects:
                        if len(group) == 1:
                            merged_rects.append(group[0])
                        else:

                            x0 = min(r.x0 for r in group)
                            y0 = min(r.y0 for r in group)
                            x1 = max(r.x1 for r in group)
                            y1 = max(r.y1 for r in group)
                            merged_rects.append(fitz.Rect(x0, y0, x1, y1))

                    for rect in merged_rects:
                        page.add_redact_annot(rect, fill=(1, 1, 1))

                    page.apply_redactions()
                    print(f"Applied {len(merged_rects)} redaction areas on page {page_num + 1}")

                    for repl in replacements:
                        if repl.get('is_position', False):

                            value = repl['new_text']
                            x = repl['text_origin'][0]
                            y = repl['text_origin'][1]
                            font_size = repl['font_size']
                            alignment = repl.get('alignment', 'left')
                            max_width = repl.get('max_width', 200)

                            if alignment == 'center':
                                page.insert_text(
                                    (x + max_width / 2, y),
                                    value,
                                    fontname="helv",
                                    fontsize=font_size,
                                    align=1
                                )
                            elif alignment == 'right':
                                page.insert_text(
                                    (x + max_width, y),
                                    value,
                                    fontname="helv",
                                    fontsize=font_size,
                                    align=2
                                )
                            else:
                                page.insert_text(
                                    (x, y),
                                    value,
                                    fontsize=font_size,
                                    fontname="helv"
                                )
                        else:

                            x, y = repl['text_origin']
                            new_text = repl['new_text']
                            font_size = repl['font_size']

                            page.insert_text(
                                point=(x, y + font_size * 0.8),
                                text=new_text,
                                fontname="helv",
                                fontsize=font_size
                            )

                        replacement_count += 1

                doc.save(output_pdf_path)
                doc.close()

                # Remove the temporary file
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {e}")

                print(f"Successfully replaced {replacement_count} placeholders")
                print(f"Filled PDF saved to: {output_pdf_path}")

            except Exception as e:
                print(f"Error during PDF placeholder replacement: {e}")
                traceback.print_exc()
                raise

    def _detect_form_fields(self, page):
            """
            Detect form fields on a page based on visual characteristics.

            Args:
                page: PDF page object

            Returns:
                List of rectangles representing form fields
            """
            form_fields = []

            # Method 1: Look for underscores that might indicate input fields
            text = page.get_text("dict")
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content = span["text"]
                        # Check for form field indicators
                        if "_" * 3 in text_content or "□" in text_content or "■" in text_content:
                            form_fields.append(fitz.Rect(span["bbox"]))
                        # Check for placeholders in parentheses
                        if "(" in text_content and ")" in text_content:
                            placeholder_pattern = r'\((.*?)\)'
                            if re.search(placeholder_pattern, text_content):
                                form_fields.append(fitz.Rect(span["bbox"]))

            # Method 2: Look for rectangles that might be form field boxes
            # This uses the page's drawing content to identify boxes
            paths = page.get_drawings()
            for path in paths:
                for item in path["items"]:
                    if item[0] == "re":  # Rectangle
                        rect = item[1]  # Rectangle coordinates
                        # Filter for rectangles that are likely to be form fields
                        # These are usually wider than they are tall and not too big
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        if 20 < width < 300 and 5 < height < 40:
                            form_fields.append(fitz.Rect(rect))

            # Method 3: Look for areas around placeholder text
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content = span["text"]
                        if "(" in text_content and ")" in text_content:
                            span_rect = fitz.Rect(span["bbox"])
                            # Expand the rectangle a bit to ensure we cover the full field
                            expanded_rect = fitz.Rect(
                                span_rect.x0 - 5,
                                span_rect.y0 - 2,
                                span_rect.x1 + 5,
                                span_rect.y1 + 2
                            )
                            form_fields.append(expanded_rect)

            return form_fields

    def _rect_overlap(self, rect1, rect2):
            """
            Check if two rectangles overlap.

            Args:
                rect1: First rectangle
                rect2: Second rectangle

            Returns:
                Boolean indicating if the rectangles overlap
            """
            # Two rectangles overlap if one rectangle's left edge is to the left of the other's right edge,
            # and one rectangle's top edge is above the other's bottom edge.
            return (
                    rect1.x0 < rect2.x1 and
                    rect1.x1 > rect2.x0 and
                    rect1.y0 < rect2.y1 and
                    rect1.y1 > rect2.y0
            )

class ImportantOutputInfo(BaseModel):
        """Defines the expected structure for output information"""
        model_config = ConfigDict(extra='forbid')

        filled_fields: List[Dict[str, Any]] = []
        failed_fields: List[Dict[str, Any]] = []
        processing_time: float = 0.0
        ocr_results_count: int = 0
        field_matches_count: int = 0
        placeholder_count: int = 0

        @field_validator('filled_fields', 'failed_fields')
        def validate_fields(cls, v):
            """Validate field lists"""
            for field in v:







                if not isinstance(field, dict):
                    raise ValueError("Each field must be a dictionary")
            return v

async def main():
        """
        Main function for PDF filling workflow with detailed logging
        """
        print("Starting FillerGEN workflow...")

        try:
            # Example usage paths
            input_pdf_path = "D:\\demo\\Services\\MIchiganCorp.pdf"
            output_pdf_path = "D:\\demo\\Services\\fill_smart5.pdf"
            json_data_path = "D:\\demo\\Services\\form_data.json"

            # Check files exist
            if not os.path.exists(input_pdf_path):
                print(f"Error: Input PDF not found at {input_pdf_path}")
                return

            if not os.path.exists(json_data_path):
                print(f"Error: JSON data not found at {json_data_path}")
                return

            # Load JSON data
            with open(json_data_path, 'r') as f:
                json_data = json.load(f)

            # Initialize filler
            filler = PDFSmartFiller()

            # Process PDF
            await filler.process_pdf(
                input_pdf_path=input_pdf_path,
                json_data=json_data,
                output_pdf_path=output_pdf_path
            )

            print("FillerGEN workflow completed successfully")

        except Exception as e:
            print(f"Error in FillerGEN workflow: {e}")

if __name__ == "__main__":
        asyncio.run(main())