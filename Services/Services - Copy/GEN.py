import asyncio
import json
import os
import re
import numpy as np
import cv2
import fitz
import time
from typing import Dict, Any, List, Tuple, Optional

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
    "field_matcher": API_KEY_3,
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


class FieldMatchingAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initialize AI agent for intelligent field matching with enhanced error handling

        Args:
            model_name (str): AI model to use
        """
        print(f"Initializing Field Matching Agent with model: {model_name}")
        try:
            self.agent = Agent(
                model=GeminiModel(model_name, api_key=API_KEYS.get("field_matcher")),
                system_prompt="""
                You are an advanced AI assistant for intelligent PDF form field mapping.

                CRITICAL OUTPUT REQUIREMENTS:
                1. ALWAYS return a VALID JSON with this EXACT structure:
                {
                    "matches": [
                        {
                            "json_field": "exact_json_key",
                            "pdf_field": "PDF field label",
                            "suggested_value": "value to fill",
                            "confidence": 0.7,
                            "reasoning": "Mapping explanation"
                        }
                    ]
                }

                2. If NO matches are found, return:
                {"matches": []}

                3. NEVER include code blocks, comments, or additional text
                """
            )
            print("Field Matching Agent initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Field Matching Agent: {e}")
            raise

    async def intelligent_field_mapping(
            self,
            json_data: Dict[str, Any],
            ocr_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Advanced field mapping using AI with comprehensive context and robust error handling

        Args:
            json_data (Dict[str, Any]): Input JSON data
            ocr_results (List[Dict[str, Any]]): OCR extracted text

        Returns:
            List[Dict[str, Any]]: Intelligent field mappings
        """
        print("Starting intelligent field mapping...")
        start_time = time.time()

        # Use a truncated version of OCR results to avoid overwhelming the model
        max_ocr_results = 100
        truncated_ocr = ocr_results[:max_ocr_results] if len(ocr_results) > max_ocr_results else ocr_results
        print(f"Using {len(truncated_ocr)} OCR results out of {len(ocr_results)}")

        # Prepare comprehensive mapping prompt
        prompt = json.dumps({
            "json_data": json_data,
            "ocr_results": truncated_ocr,
            "mapping_instructions": {
                "prioritize_semantic_matching": True,
                "handle_nested_structures": True,
                "suggest_value_transformations": True
            }
        }, indent=2)

        try:
            print("Sending request to AI model...")
            # Add timeout to the API call
            response_future = self.agent.run(prompt)
            response = await asyncio.wait_for(response_future, timeout=API_TIMEOUT)
            print(f"Received AI response in {time.time() - start_time:.2f} seconds")
            return self._parse_and_validate_matches(response.data)

        except asyncio.TimeoutError:
            print(f"AI request timed out after {API_TIMEOUT} seconds")
            return []
        except Exception as e:
            print(f"Field mapping error: {e}")
            return []

    def _parse_and_validate_matches(self, raw_response: str) -> List[Dict[str, Any]]:
        """
        Comprehensive parsing and validation of AI-generated field matches

        Args:
            raw_response (str): Raw AI response

        Returns:
            List[Dict[str, Any]]: Validated field matches

        Raises:
            AIResponseValidationError: If response cannot be parsed or validated
        """
        print("Parsing AI response...")

        # Extensive cleaning and preprocessing
        try:
            # Remove any code block markers, trim whitespace
            clean_response = re.sub(r'```(json)?', '', raw_response).strip()
            clean_response = clean_response.replace('\n', '').replace('\r', '')

            print(f"Cleaned response length: {len(clean_response)} characters")
            print(f"Response preview: {clean_response[:100]}...")

            # Attempt to parse JSON
            try:
                parsed_matches = json.loads(clean_response)
            except json.JSONDecodeError as json_err:
                print(f"JSON Decoding Error: {json_err}")
                print(f"First 200 chars of problematic response: {clean_response[:200]}")
                raise AIResponseValidationError("Invalid JSON structure")

            # Validate parsed matches
            validated_matches = []
            match_count = len(parsed_matches.get('matches', []))
            print(f"Found {match_count} raw matches to validate")

            for match in parsed_matches.get('matches', []):
                # Rigorous validation of match structure
                if not all(key in match for key in ['json_field', 'pdf_field', 'suggested_value']):
                    print(f"Incomplete match: {match}")
                    continue

                # Normalize and validate match
                validated_match = {
                    'json_field': str(match['json_field']),
                    'pdf_field': str(match['pdf_field']),
                    'suggested_value': str(match['suggested_value']),
                    'confidence': float(match.get('confidence', 0.7)),
                    'reasoning': str(match.get('reasoning', 'No specific reasoning'))
                }

                validated_matches.append(validated_match)

            print(f"Parsed {len(validated_matches)} valid field matches")
            return validated_matches

        except Exception as e:
            print(f"Comprehensive parsing error: {e}")
            print(f"First 200 chars of original response: {raw_response[:200]}")
            raise AIResponseValidationError(f"Failed to parse AI response: {e}")


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
                You are an AI specialized in analyzing PDF forms and determining the exact positions 
                where field values should be placed.

                For each form field, you need to identify:
                1. The page number where the field is located
                2. The precise coordinates where text should be inserted
                3. Any special formatting requirements

                CRITICAL OUTPUT REQUIREMENTS:
                ALWAYS return a VALID JSON with this EXACT structure:
                {
                    "field_positions": [
                        {
                            "field_name": "Field name as it appears in PDF",
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

    async def analyze_form_structure(
            self,
            ocr_results: List[Dict[str, Any]],
            pdf_width: int,
            pdf_height: int,
            field_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze form structure to determine optimal field placement

        Args:
            ocr_results: OCR results from the PDF
            pdf_width: Width of the PDF in points
            pdf_height: Height of the PDF in points
            field_matches: List of field matches to place in the PDF

        Returns:
            List of field position mappings with exact coordinates
        """
        print("Starting form structure analysis...")
        start_time = time.time()

        # Use a truncated version of OCR results to avoid overwhelming the model
        max_ocr_results = 100
        truncated_ocr = ocr_results[:max_ocr_results] if len(ocr_results) > max_ocr_results else ocr_results
        print(f"Using {len(truncated_ocr)} OCR results out of {len(ocr_results)}")

        # Prepare the fields to place - limit to 20 fields max to avoid overwhelming the model
        fields_to_place = [match['pdf_field'] for match in field_matches][:20]
        print(f"Analyzing placement for {len(fields_to_place)} fields")

        # Prepare the analysis prompt
        prompt = json.dumps({
            "ocr_results": truncated_ocr,
            "pdf_dimensions": {
                "width": pdf_width,
                "height": pdf_height
            },
            "fields_to_place": fields_to_place,
            "instructions": "Analyze the document structure and determine the exact positions where each field value should be placed. Look for form labels, underscores, boxes, or other indicators of field positions."
        }, indent=2)

        try:
            print("Sending request to AI model for form analysis...")
            # Add timeout to the API call
            response_future = self.agent.run(prompt)
            response = await asyncio.wait_for(response_future, timeout=API_TIMEOUT)
            print(f"Received AI form analysis response in {time.time() - start_time:.2f} seconds")
            return self._parse_and_validate_positions(response.data)
        except asyncio.TimeoutError:
            print(f"AI request timed out after {API_TIMEOUT} seconds")
            return []
        except Exception as e:
            print(f"Form structure analysis error: {e}")
            return []

    def _parse_and_validate_positions(self, raw_response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate the AI response for field positions

        Args:
            raw_response: Raw AI response

        Returns:
            List of validated field positions
        """
        print("Parsing field position analysis...")

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

            # Extract and validate field positions
            field_positions = []
            position_count = len(parsed_positions.get('field_positions', []))
            print(f"Found {position_count} raw positions to validate")

            for position in parsed_positions.get('field_positions', []):
                if not all(key in position for key in ['field_name', 'page', 'coordinates']):
                    print(f"Incomplete position data: {position}")
                    continue

                field_positions.append(position)

            print(f"Parsed {len(field_positions)} valid field positions")
            return field_positions

        except Exception as e:
            print(f"Position parsing error: {e}")
            return []


class PDFSmartFiller:
    def __init__(self):
        print("Initializing PDF Smart Filler...")
        self.ocr_processor = AdvancedOCRProcessor()
        self.field_matcher = FieldMatchingAgent()
        self.field_analyzer = FormFieldAnalyzer()
        print("PDF Smart Filler initialized")

    async def process_pdf(
            self,
            input_pdf_path: str,
            json_data: Dict[str, Any],
            output_pdf_path: str
    ):
        """
        Comprehensive PDF processing with advanced OCR and AI mapping

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

            # Convert PDF to images with progress reporting
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

                # Break early if processing is taking too long
                if time.time() - ocr_start_time > OCR_TIMEOUT:
                    print(
                        f"OCR taking too long (> {OCR_TIMEOUT} seconds). Processing only the first {page_num + 1} pages.")
                    break

            ocr_time = time.time() - ocr_start_time
            print(f"OCR completed in {ocr_time:.2f} seconds. Found {len(all_ocr_results)} total text elements.")

            # Intelligent field mapping
            try:
                print("Starting field mapping...")
                mapping_start_time = time.time()

                field_matches = await self.field_matcher.intelligent_field_mapping(
                    json_data,
                    all_ocr_results
                )

                mapping_time = time.time() - mapping_start_time
                print(f"Field mapping completed in {mapping_time:.2f} seconds. Found {len(field_matches)} matches.")

                # Ensure we have matches before proceeding
                if not field_matches:
                    print("No field matches found. Skipping PDF filling.")
                    return

                # Get PDF dimensions for AI analysis
                print("Analyzing PDF structure...")
                doc = fitz.open(input_pdf_path)
                first_page = doc[0]
                pdf_width, pdf_height = first_page.rect.width, first_page.rect.height
                print(f"PDF dimensions: {pdf_width}x{pdf_height}")

                # Analyze form structure to determine field positions
                analysis_start_time = time.time()
                field_positions = await self.field_analyzer.analyze_form_structure(
                    all_ocr_results,
                    pdf_width,
                    pdf_height,
                    field_matches
                )

                analysis_time = time.time() - analysis_start_time
                print(
                    f"Form structure analysis completed in {analysis_time:.2f} seconds. Found {len(field_positions)} field positions.")

                # Fill PDF using the determined positions
                filling_start_time = time.time()
                await self._fill_pdf_with_ai_positions(
                    input_pdf_path,
                    output_pdf_path,
                    field_matches,
                    field_positions
                )

                filling_time = time.time() - filling_start_time
                print(f"PDF filling completed in {filling_time:.2f} seconds.")

            except AIResponseValidationError as ave:
                print(f"AI Response Validation Failed: {ave}")

            overall_time = time.time() - overall_start_time
            print(f"Overall PDF processing completed in {overall_time:.2f} seconds.")
            print(f"Output saved to: {output_pdf_path}")

        except Exception as e:
            print(f"PDF processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _fill_pdf_with_ai_positions(
            self,
            input_pdf_path: str,
            output_pdf_path: str,
            field_matches: List[Dict[str, Any]],
            field_positions: List[Dict[str, Any]]
    ):
        """
        Fill PDF using AI-determined field positions

        Args:
            input_pdf_path: Input PDF path
            output_pdf_path: Output PDF path
            field_matches: Field match data
            field_positions: AI-determined field positions
        """
        print(f"Opening PDF for filling: {input_pdf_path}")
        doc = fitz.open(input_pdf_path)

        # Check if the PDF has actual form fields
        has_form_fields = False
        form_field_count = 0
        for page in doc:
            widgets = page.widgets()
            # Convert to list if it's a generator
            if hasattr(widgets, '__iter__') and not hasattr(widgets, '__len__'):
                widgets = list(widgets)
            form_field_count += len(widgets)
            if widgets:
                has_form_fields = True

        if has_form_fields:
            # Method 1: Fill actual form fields if they exist
            print(f"PDF has {form_field_count} form fields. Filling those directly.")
            self._fill_form_fields(doc, field_matches)
        else:
            # Method 2: Use AI-determined positions
            print("Using AI-determined positions to place text")
            self._place_text_with_ai_positions(doc, field_matches, field_positions)

            # If AI didn't provide enough positions, fall back to a backup method
            if len(field_positions) < len(field_matches):
                print(f"AI provided positions for only {len(field_positions)} out of {len(field_matches)} fields")
                print("Using backup method for remaining fields")
                self._place_text_with_backup_method(doc, field_matches, field_positions)

        print(f"Saving filled PDF to: {output_pdf_path}")
        doc.save(output_pdf_path)
        print(f"Filled PDF saved: {output_pdf_path}")

    def _fill_form_fields(self, doc, field_matches):
        """Fill actual PDF form fields"""
        print("Starting to fill form fields...")
        fields_filled = 0

        for match in field_matches:
            match_found = False
            for page in doc:
                fields = page.widgets()
                # Convert to list if it's a generator
                if hasattr(fields, '__iter__') and not hasattr(fields, '__len__'):
                    fields = list(fields)

                for field in fields:
                    field_name = field.field_name if hasattr(field, 'field_name') else ""
                    if field_name.lower() == match['pdf_field'].lower() or match[
                        'pdf_field'].lower() in field_name.lower():
                        field.text_font = "Helvetica"
                        field.text_fontsize = 10
                        field.field_value = str(match['suggested_value'])
                        field.update()
                        fields_filled += 1
                        match_found = True
                        print(f"Filled form field '{field_name}' with '{match['suggested_value']}'")

            if not match_found:
                print(f"No form field found for '{match['pdf_field']}'")

        print(f"Filled {fields_filled} form fields out of {len(field_matches)} matches")

    def _place_text_with_ai_positions(self, doc, field_matches, field_positions):
        """Place text using AI-determined positions"""
        print("Placing text using AI-determined positions...")
        fields_placed = 0

        # Create position lookup by field name
        position_lookup = {pos['field_name']: pos for pos in field_positions}

        for match in field_matches:
            pdf_field = match['pdf_field']
            if pdf_field in position_lookup:
                pos = position_lookup[pdf_field]
                page_num = pos['page']

                if page_num < len(doc):
                    page = doc[page_num]

                    # Get coordinates
                    x = pos['coordinates']['x']
                    y = pos['coordinates']['y']

                    # Get formatting options with defaults
                    formatting = pos.get('formatting', {})
                    font_size = formatting.get('font_size', 10)
                    alignment = formatting.get('alignment', 'left')
                    max_width = pos.get('max_width', 200)

                    # Determine alignment
                    align_val = 0  # left
                    if alignment == 'center':
                        align_val = 1
                    elif alignment == 'right':
                        align_val = 2

                    # Create a rectangle for text placement
                    text_rect = fitz.Rect(x, y, x + max_width, y + font_size * 1.5)

                    # Insert textbox with the value
                    page.insert_textbox(
                        text_rect,
                        str(match['suggested_value']),
                        fontname="Helvetica",
                        fontsize=font_size,
                        align=align_val
                    )

                    fields_placed += 1
                    print(f"Placed '{match['suggested_value']}' for field '{pdf_field}' on page {page_num + 1}")

        print(f"Placed {fields_placed} fields using AI-determined positions")

    def _place_text_with_backup_method(self, doc, field_matches, field_positions):
        """Backup method for placing text when AI positions are insufficient"""
        print("Using backup method for positioning fields...")
        fields_placed = 0

        # Create a set of fields that already have positions
        fields_with_positions = {pos['field_name'] for pos in field_positions}

        # Process only fields without positions
        remaining_matches = [match for match in field_matches if match['pdf_field'] not in fields_with_positions]
        print(f"Attempting to place {len(remaining_matches)} remaining fields")

        # Create a simple approximation of form field locations
        for match in remaining_matches:
            pdf_field = match['pdf_field']
            field_placed = False

            for page_num, page in enumerate(doc):
                # Search for the field label text
                text_instances = page.search_for(pdf_field)
                placeholder_instances = page.search_for("___")

                if text_instances:
                    # Found the field label, place text to the right
                    rect = text_instances[0]
                    page.insert_text(
                        (rect.x1 + 5, rect.y0 + 12),  # position slightly to the right and aligned with top
                        str(match['suggested_value']),
                        fontname="Helvetica",
                        fontsize=10
                    )
                    fields_placed += 1
                    field_placed = True
                    print(
                        f"Backup method: Placed '{match['suggested_value']}' for field '{pdf_field}' on page {page_num + 1}")
                    break
                elif placeholder_instances and not field_placed:
                    # Found a placeholder, use it
                    for rect in placeholder_instances:
                        # Check if this placeholder is near any OCR text that might match our field
                        ocr_matches = page.get_text("words")
                        for word in ocr_matches:
                            word_text = word[4]
                            word_rect = fitz.Rect(word[0], word[1], word[2], word[3])

                            # If the word is close to the placeholder and might be related to our field
                            if (word_rect.distance_to(rect) < 50 and
                                    (pdf_field.lower() in word_text.lower() or
                                     any(token.lower() in word_text.lower() for token in pdf_field.lower().split()))):
                                page.insert_text(
                                    (rect.x0 + 2, rect.y0 + 12),
                                    str(match['suggested_value']),
                                    fontname="Helvetica",
                                    fontsize=10
                                )
                                fields_placed += 1
                                field_placed = True
                                print(
                                    f"Backup method: Placed '{match['suggested_value']}' near underscore for '{pdf_field}' on page {page_num + 1}")
                                break

                        if field_placed:
                            break

                    if field_placed:
                        break

            if not field_placed:
                print(f"Could not find a position for field '{pdf_field}'")

        print(f"Placed {fields_placed} fields using backup method")


async def main():
    try:
        print("Program started")
        print("Initializing Smart PDF Filler...")

        # Create smart filler instance
        smart_filler = PDFSmartFiller()

        # Define paths
        input_pdf = "D:\\demo\\Services\\MIchiganCorp.pdf"
        json_path = "D:\\demo\\Services\\form_data.json"
        output_pdf = "D:\\demo\\Services\\fill_smart_final.pdf"

        print(f"Input PDF: {input_pdf}")
        print(f"JSON data: {json_path}")
        print(f"Output PDF: {output_pdf}")

        # Verify paths exist
        if not os.path.exists(input_pdf):
            print(f"Error: Input PDF not found: {input_pdf}")
            return

        if not os.path.exists(json_path):
            print(f"Error: JSON data file not found: {json_path}")
            return

        # Load JSON data with error handling
        try:
            print(f"Loading JSON data from: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load JSON data: {e}")
            return

        # Process PDF
        await smart_filler.process_pdf(input_pdf, json_data, output_pdf)

    except Exception as e:
        print(f"Unexpected error in main process: {e}")


if __name__ == "__main__":
    asyncio.run(main())