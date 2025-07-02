# import fitz
# from typing import Dict, Any, List, Optional, Tuple
# import logging
# import os
# import json
# import base64
# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import dataclass
# from functools import lru_cache
# import io
# import time
# import google.generativeai as genai
# from PIL import Image
# import numpy as np
# from Common.constants import *
# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class PDFField:
#     """Dataclass to represent a PDF form field with all relevant metadata."""
#     name: str
#     page_num: int
#     field_type: str
#     rect: List[float]
#     is_readonly: bool
#     value: Any
#     options: Optional[List[str]] = None
#     is_required: bool = False
#     tooltip: Optional[str] = None
#     label: Optional[str] = None  # Field label detected by AI
#     field_type_name: str = ""
#
#     def __post_init__(self):
#         """Convert numeric field type to readable string."""
#         type_map = {
#             1: "TEXT",
#             2: "CHECKBOX",
#             3: "RADIO",
#             4: "LISTBOX",
#             5: "COMBOBOX",
#             6: "SIGNATURE"
#         }
#         self.field_type_name = type_map.get(self.field_type, f"UNKNOWN({self.field_type})")
#
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert field to dictionary representation."""
#         return {
#             "name": self.name,
#             "page_num": self.page_num,
#             "type": self.field_type_name,
#             "rect": self.rect,
#             "is_readonly": self.is_readonly,
#             "is_required": self.is_required,
#             "value": self.value,
#             "options": self.options,
#             "tooltip": self.tooltip,
#             "label": self.label
#         }
#
#
# class GeminiPDFExtractor:
#     """PDF form field extractor using Gemini for enhanced extraction and analysis."""
#
#     def __init__(self,
#                  gemini_api_key: str,
#                  model_name: str = "gemini-1.5-pro-vision",
#                  cache_size: int = 128,
#                  enable_ai: bool = True):
#         """Initialize the extractor with Gemini API credentials and settings."""
#         self.gemini_api_key = gemini_api_key
#         self.model_name = model_name
#         self.enable_ai = enable_ai
#
#         # Configure Gemini
#         if self.gemini_api_key and self.enable_ai:
#             genai.configure(api_key=self.gemini_api_key)
#
#         # Use LRU cache for repeated processing
#         self.extract_pdf_fields = lru_cache(maxsize=cache_size)(self._extract_pdf_fields)
#
#     def _extract_basic_fields(self, doc: fitz.Document) -> Dict[str, PDFField]:
#         """Extract basic field information using PyMuPDF."""
#         fields = {}
#
#         for page_num, page in enumerate(doc):
#             try:
#                 for widget in page.widgets():
#                     if not widget.field_name:
#                         continue
#
#                     field_name = widget.field_name.strip()
#                     is_required = bool(widget.field_flags & 2)
#                     tooltip = getattr(widget, "tooltip", None)
#
#                     options = []
#                     if widget.field_type in (4, 5):  # LISTBOX or COMBOBOX
#                         try:
#                             options = widget.choice_values
#                         except Exception:
#                             pass
#
#                     try:
#                         if widget.field_type == 2:  # CHECKBOX
#                             value = bool(widget.field_value)
#                         else:
#                             value = widget.field_value
#                     except Exception:
#                         value = None
#
#                     field = PDFField(
#                         name=field_name,
#                         page_num=page_num,
#                         field_type=widget.field_type,
#                         rect=[widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1],
#                         is_readonly=bool(widget.field_flags & 1),
#                         value=value,
#                         options=options,
#                         is_required=is_required,
#                         tooltip=tooltip
#                     )
#
#                     fields[field_name] = field
#             except Exception as e:
#                 logger.error(f"Error processing page {page_num}: {e}")
#
#         return fields
#
#     def _get_page_image(self, page: fitz.Page, scale: float = 1.5) -> Image.Image:
#         """Render a PDF page to a PIL Image."""
#         pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
#         img_bytes = io.BytesIO()
#         pix.save(img_bytes, "png")
#         img_bytes.seek(0)
#         return Image.open(img_bytes)
#
#     def _analyze_page_with_gemini(self, page: fitz.Page, form_fields: List[Dict]) -> Dict[str, Any]:
#         """Use Gemini to analyze page content and enhance field information."""
#         if not self.gemini_api_key or not self.enable_ai:
#             return {}
#
#         try:
#             # Get page as image
#             img = self._get_page_image(page)
#
#             # Initialize Gemini model
#             model = genai.GenerativeModel(self.model_name)
#
#             # Prepare prompt
#             prompt = f"""
#             I'm analyzing a PDF form with the following form fields on this page:
#             {json.dumps(form_fields, indent=2)}
#
#             For each form field, please:
#             1. Identify any text labels near the field (considering proximity and alignment)
#             2. Classify what information this field is likely collecting based on context
#             3. Suggest a user-friendly name if the current field name is cryptic
#             4. Indicate if this field appears to be part of a logical group with other fields
#
#             Return your analysis as a JSON object where each key is the original field name.
#             The JSON should be valid and parseable.
#             """
#
#             # Make API request with retry logic
#             for attempt in range(3):
#                 try:
#                     response = model.generate_content([prompt, img])
#                     break
#                 except Exception as e:
#                     if attempt == 2:
#                         raise
#                     logger.warning(f"Gemini API request failed (attempt {attempt + 1}): {e}. Retrying...")
#                     time.sleep(2)
#
#             # Extract text from response
#             response_text = response.text
#
#             # Try to extract JSON from the response
#             try:
#                 # First attempt: try to parse the entire response as JSON
#                 return json.loads(response_text)
#             except json.JSONDecodeError:
#                 # Second attempt: Look for JSON code block
#                 import re
#                 json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
#                 if json_match:
#                     try:
#                         return json.loads(json_match.group(1))
#                     except json.JSONDecodeError:
#                         logger.error("Failed to parse JSON from code block")
#
#                 # Third attempt: Look for any JSON-like structure
#                 json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
#                 if json_match:
#                     try:
#                         return json.loads(json_match.group(1))
#                     except json.JSONDecodeError:
#                         logger.error("Failed to parse JSON-like structure")
#
#                 logger.error(f"Failed to extract JSON from Gemini response")
#                 return {}
#
#         except Exception as e:
#             logger.error(f"Gemini analysis failed: {e}")
#             return {}
#
#     def _enhance_fields_with_gemini(self, doc: fitz.Document, fields: Dict[str, PDFField]) -> Dict[str, PDFField]:
#         """Enhance field information using Gemini for each page."""
#         if not self.gemini_api_key or not self.enable_ai or not fields:
#             return fields
#
#         # Process page by page
#         for page_num in range(len(doc)):
#             # Get fields on this page
#             page_fields = {k: v for k, v in fields.items() if v.page_num == page_num}
#             if not page_fields:
#                 continue
#
#             # Prepare basic field info for Gemini
#             field_info = [{
#                 "name": field.name,
#                 "type": field.field_type_name,
#                 "rect": field.rect
#             } for field in page_fields.values()]
#
#             # Analyze with Gemini
#             gemini_results = self._analyze_page_with_gemini(doc[page_num], field_info)
#
#             # Enhance fields with Gemini insights
#             for field_name, ai_data in gemini_results.items():
#                 if field_name in fields:
#                     # Extract label if detected
#                     if "label" in ai_data:
#                         fields[field_name].label = ai_data["label"]
#
#                     # Update other metadata as needed
#                     if "user_friendly_name" in ai_data and ai_data["user_friendly_name"]:
#                         logger.info(f"Gemini suggested name for '{field_name}': {ai_data['user_friendly_name']}")
#
#         return fields
#
#     def _extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
#         """Extract all fillable fields with Gemini enhancement."""
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
#
#         logger.info(f"Extracting fields from: {pdf_path}")
#
#         try:
#             doc = fitz.open(pdf_path)
#
#             # Extract basic field info
#             fields = self._extract_basic_fields(doc)
#             logger.info(f"Basic extraction found {len(fields)} fields")
#
#             # Enhance with Gemini if enabled
#             if self.enable_ai and self.gemini_api_key:
#                 logger.info("Enhancing fields with Gemini analysis...")
#                 fields = self._enhance_fields_with_gemini(doc, fields)
#
#             # Convert to dictionary representation
#             results = {name: field.to_dict() for name, field in fields.items()}
#
#             doc.close()
#
#             # Log summary
#             for field, info in results.items():
#                 status = []
#                 if info["is_readonly"]:
#                     status.append("READ-ONLY")
#                 if info["is_required"]:
#                     status.append("REQUIRED")
#                 status_str = ", ".join(status) if status else "EDITABLE"
#
#                 label_info = f" - Label: '{info['label']}'" if info.get("label") else ""
#                 logger.info(
#                     f"Field: '{field}' (Page {info['page_num'] + 1}) [{info['type']}] [{status_str}]{label_info}")
#
#             return results
#
#         except Exception as e:
#             logger.error(f"Failed to extract fields from {pdf_path}: {e}")
#             raise
#
#     def predict_form_completion(self, pdf_path: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Use Gemini to predict the most appropriate values for unfilled form fields."""
#         if not self.gemini_api_key or not self.enable_ai:
#             return {}
#
#         try:
#             # Get existing fields and their current values
#             fields = self.extract_pdf_fields(pdf_path)
#
#             # Prepare the contextual information for Gemini
#             context = {
#                 "filled_fields": {k: v["value"] for k, v in fields.items() if v["value"]},
#                 "unfilled_fields": [k for k, v in fields.items() if not v["value"]],
#                 "provided_data": form_data
#             }
#
#             # Initialize Gemini text model
#             text_model = genai.GenerativeModel("gemini-1.5-pro")
#
#             # Create prompt
#             prompt = f"""
#             Based on the context of already filled fields and the provided data, suggest appropriate
#             values for the unfilled fields in this form. Return a JSON object mapping field names to
#             suggested values. Only include fields that can be confidently filled.
#
#             Form context: {json.dumps(context, indent=2)}
#
#             Return only valid JSON.
#             """
#
#             # Make request to Gemini
#             response = text_model.generate_content(prompt)
#             response_text = response.text
#
#             # Parse JSON from response
#             try:
#                 # First attempt with direct parsing
#                 return json.loads(response_text)
#             except json.JSONDecodeError:
#                 # Try to extract JSON from text
#                 import re
#                 json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
#
#                 # One more attempt with any JSON-like structure
#                 json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
#
#                 logger.error("Failed to extract valid JSON from Gemini response")
#                 return {}
#
#         except Exception as e:
#             logger.error(f"Field prediction failed: {e}")
#             return {}
#
#     def detect_field_groups(self, pdf_path: str) -> Dict[str, List[str]]:
#         """Detect logical groupings of form fields using Gemini."""
#         if not self.gemini_api_key or not self.enable_ai:
#             return {}
#
#         fields = self.extract_pdf_fields(pdf_path)
#
#         try:
#             # Initialize Gemini text model
#             text_model = genai.GenerativeModel("gemini-1.5-pro")
#
#             # Create prompt
#             prompt = f"""
#             Analyze these form fields and identify logical groupings based on field names, types, and page locations.
#             Examples of logical groups: "Personal Information", "Address", "Payment Details", etc.
#             Return a JSON object where keys are group names and values are lists of field names.
#
#             Form fields: {json.dumps(fields, indent=2)}
#
#             Return only valid JSON.
#             """
#
#             # Make request to Gemini
#             response = text_model.generate_content(prompt)
#             response_text = response.text
#
#             # Parse JSON from response
#             try:
#                 return json.loads(response_text)
#             except json.JSONDecodeError:
#                 # Try to extract JSON from markdown code block
#                 import re
#                 json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
#
#                 # One more attempt with any JSON-like structure
#                 json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
#
#                 logger.error("Failed to extract valid JSON from Gemini response")
#                 return {}
#
#         except Exception as e:
#             logger.error(f"Field grouping detection failed: {e}")
#             return {}
#
#     def extract_fields_batch(self, pdf_paths: List[str], max_workers: int = 4) -> Dict[str, Dict[str, Dict[str, Any]]]:
#         """Process multiple PDFs in parallel using ThreadPoolExecutor."""
#         results = {}
#
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_pdf = {executor.submit(self.extract_pdf_fields, path): path for path in pdf_paths}
#
#             for future in future_to_pdf:
#                 pdf_path = future_to_pdf[future]
#                 try:
#                     results[pdf_path] = future.result()
#                 except Exception as e:
#                     logger.error(f"Failed to process {pdf_path}: {e}")
#                     results[pdf_path] = {"error": str(e)}
#
#         return results
#
#     def analyze_form_structure(self, pdf_path: str) -> Dict[str, Any]:
#         """Use Gemini to analyze the overall structure and purpose of the form."""
#         if not self.gemini_api_key or not self.enable_ai:
#             return {}
#
#         try:
#             doc = fitz.open(pdf_path)
#             first_page_img = self._get_page_image(doc[0])
#
#             # Initialize Gemini model
#             model = genai.GenerativeModel(self.model_name)
#
#             # Create prompt
#             prompt = """
#             Analyze this PDF form and provide the following information:
#             1. What type of form is this? (e.g., medical intake, job application, tax form)
#             2. Who is the likely issuer of this form?
#             3. What is the primary purpose of this form?
#             4. What categories of information does this form collect?
#             5. Are there any compliance or regulatory aspects visible in this form?
#
#             Return your analysis as a JSON object with these categories.
#             """
#
#             # Make request to Gemini
#             response = model.generate_content([prompt, first_page_img])
#             response_text = response.text
#
#             # Try to parse JSON
#             try:
#                 return json.loads(response_text)
#             except json.JSONDecodeError:
#                 # Try to find JSON in the response
#                 import re
#                 json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group(1))
#
#                 # Return structured data even if not JSON
#                 logger.warning("Could not extract JSON from form structure analysis")
#                 return {"analysis": response_text}
#
#         except Exception as e:
#             logger.error(f"Form structure analysis failed: {e}")
#             return {}
#         finally:
#             if 'doc' in locals():
#                 doc.close()
#
#     def analyze_field_positions(self, pdf_path: str) -> Dict[str, Any]:
#         """Generate heat maps and positional analysis of form fields using extracted field positions."""
#         try:
#             fields = self.extract_pdf_fields(pdf_path)
#
#             # Open document to get page dimensions
#             doc = fitz.open(pdf_path)
#             page_dimensions = []
#             for i in range(len(doc)):
#                 page = doc[i]
#                 page_dimensions.append((page.rect.width, page.rect.height))
#             doc.close()
#
#             # Group fields by page
#             fields_by_page = {}
#             for field_name, field_data in fields.items():
#                 page_num = field_data["page_num"]
#                 if page_num not in fields_by_page:
#                     fields_by_page[page_num] = []
#                 fields_by_page[page_num].append({
#                     "name": field_name,
#                     "rect": field_data["rect"],
#                     "type": field_data["type"]
#                 })
#
#             # Create position analysis
#             position_analysis = {}
#             for page_num, page_fields in fields_by_page.items():
#                 if page_num >= len(page_dimensions):
#                     continue
#
#                 width, height = page_dimensions[page_num]
#
#                 # Create empty heatmap data (10x10 grid)
#                 grid_size = 10
#                 heatmap = np.zeros((grid_size, grid_size))
#
#                 # Map fields to grid
#                 for field in page_fields:
#                     x0, y0, x1, y1 = field["rect"]
#
#                     # Convert to grid coordinates
#                     grid_x0 = min(int((x0 / width) * grid_size), grid_size - 1)
#                     grid_y0 = min(int((y0 / height) * grid_size), grid_size - 1)
#                     grid_x1 = min(int((x1 / width) * grid_size), grid_size - 1)
#                     grid_y1 = min(int((y1 / height) * grid_size), grid_size - 1)
#
#                     # Update heatmap
#                     for gx in range(grid_x0, grid_x1 + 1):
#                         for gy in range(grid_y0, grid_y1 + 1):
#                             heatmap[gy, gx] += 1
#
#                 # Analyze field distribution
#                 field_count = len(page_fields)
#                 top_half_count = sum(1 for field in page_fields if field["rect"][1] < height / 2)
#                 left_half_count = sum(1 for field in page_fields if field["rect"][0] < width / 2)
#
#                 position_analysis[f"page_{page_num + 1}"] = {
#                     "field_count": field_count,
#                     "top_half_percentage": (top_half_count / field_count) * 100 if field_count > 0 else 0,
#                     "left_half_percentage": (left_half_count / field_count) * 100 if field_count > 0 else 0,
#                     "heatmap": heatmap.tolist(),
#                     "field_type_distribution": {
#                         field_type: sum(1 for field in page_fields if field["type"] == field_type)
#                         for field_type in set(field["type"] for field in page_fields)
#                     }
#                 }
#
#             return position_analysis
#
#         except Exception as e:
#             logger.error(f"Field position analysis failed: {e}")
#             return {}
#
#
# # Example usage
# if __name__ == "__main__":
#     # Initialize with your Gemini API key
#     extractor = GeminiPDFExtractor(
#         gemini_api_key=API_KEY_3,
#         enable_ai=True  # Set to False to disable AI features
#     )
#
#     # Extract fields with Gemini enhancement
#     fields = extractor.extract_pdf_fields("D:\\demo\\Services\\Maine.pdf")
#     print(f"Extracted {len(fields)} fields")
#
#     # Analyze form structure
#     structure = extractor.analyze_form_structure("D:\\demo\\Services\\Maine.pdf")
#     print(f"Form type: {structure.get('form_type', 'Unknown')}")
#
#     # Detect logical field groupings
#     groups = extractor.detect_field_groups("D:\\demo\\Services\\Maine.pdf")
#     print(f"Detected {len(groups)} field groups")
#
#     # Predict values for unfilled fields
#     user_data = {
#         "name": "John Smith",
#         "email": "john.smith@example.com",
#         "address": "123 Main St, Anytown, CA 94538"
#     }
#     predictions = extractor.predict_form_completion("D:\\demo\\Services\\Maine.pdf", user_data)
#     print(f"Generated predictions for {len(predictions)} fields")
#
#     # Batch process multiple PDFs
#     batch_results = extractor.extract_fields_batch(["form1.pdf", "form2.pdf"])
#     for pdf_path, result in batch_results.items():
#         print(f"Processed {pdf_path}: {len(result)} fields")


import fitz
import logging
import os
import json
import io
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from Common.constants import  *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PDFField:
    """Dataclass to represent a PDF form field with label information."""
    name: str  # Field name in the PDF
    page_num: int
    field_type: str
    rect: List[float]
    label: str  # The human-readable label associated with this field
    label_rect: Optional[List[float]] = None  # Coordinates of the label text
    confidence: float = 1.0  # AI confidence score

    def to_dict(self) -> Dict[str, Any]:
        """Convert field to dictionary representation."""
        return {
            "name": self.name,
            "page_num": self.page_num,
            "type": self.field_type,
            "rect": self.rect,
            "label": self.label,
            "label_rect": self.label_rect,
            "confidence": self.confidence
        }


class GeminiPDFLabelExtractor:
    """PDF form field and label extractor using Gemini vision capabilities."""

    def __init__(self,
                 gemini_api_key: str,
                 model_name: str = "gemini-1.5-flash",
                 scale_factor: float = 1.5,
                 max_retries: int = 3):
        """Initialize the extractor with Gemini API credentials and settings."""
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.max_retries = max_retries

        # Configure Gemini
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        else:
            raise ValueError("Gemini API key is required")

    def _get_page_image(self, page: fitz.Page) -> Image.Image:
        """Render a PDF page to a PIL Image with appropriate scaling."""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(self.scale_factor, self.scale_factor))
            img_bytes = io.BytesIO()
            pix.pil_save(img_bytes, format="PNG")
            img_bytes.seek(0)
            return Image.open(img_bytes)
        except Exception as e:
            logger.error(f"Error rendering page image with pil_save: {e}")
            try:
                # Alternative approach
                pix = page.get_pixmap(matrix=fitz.Matrix(self.scale_factor, self.scale_factor))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return img
            except Exception as e2:
                logger.error(f"Error rendering page image with alternative method: {e2}")
                # Final fallback with lower quality
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return img

    def _extract_fields_and_labels(self, page: fitz.Page, page_num: int) -> List[PDFField]:
        """Extract form fields and their associated labels using Gemini's visual capabilities."""
        try:
            # Get page as image
            img = self._get_page_image(page)

            # Initialize Gemini model
            model = genai.GenerativeModel(self.model_name)

            # Create extraction prompt focused on labels
            prompt = """
            Analyze this PDF document image and identify all form fields present with their associated labels. 

            For each form field:
            1. Identify the field itself (text box, checkbox, radio button, dropdown, etc.)
            2. Find the associated label or text description that explains what this field is for
            3. Determine the bounding box coordinates for both the field and its label

            Your task is to correctly pair each form field with its associated label text.

            Return the results as a JSON array of objects, where each object represents one form field with these properties:
            - "name": A descriptive name for the field (can be based on the label)
            - "field_type": string (TEXT, CHECKBOX, RADIO, DROPDOWN, SIGNATURE)
            - "rect": [x0, y0, x1, y1] (field coordinates, normalized from 0 to 1)
            - "label": The exact text of the label associated with this field
            - "label_rect": [x0, y0, x1, y1] (label coordinates, normalized from 0 to 1)
            - "confidence": number from 0 to 1 indicating detection confidence

            Important: Focus on correctly identifying which label belongs to which field, paying attention to proximity and alignment.
            Return valid JSON only.
            """

            # Make request to Gemini with retry logic
            fields_data = None

            for attempt in range(self.max_retries):
                try:
                    response = model.generate_content([prompt, img])
                    response_text = response.text

                    # Try to extract JSON from the response
                    try:
                        # First attempt: direct JSON parsing
                        fields_data = json.loads(response_text)
                        break
                    except json.JSONDecodeError:
                        # Second attempt: Look for JSON code block
                        import re
                        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                        if json_match:
                            try:
                                fields_data = json.loads(json_match.group(1))
                                break
                            except json.JSONDecodeError:
                                pass

                        # Third attempt: Look for any JSON-like structure
                        json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
                        if json_match:
                            try:
                                fields_data = json.loads(json_match.group(1))
                                break
                            except json.JSONDecodeError:
                                pass

                        logger.warning(f"Could not extract valid JSON on attempt {attempt + 1}")

                except Exception as e:
                    logger.warning(f"Gemini API error on attempt {attempt + 1}: {e}")

                # Wait before retry
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff

            if not fields_data:
                logger.error("All attempts to extract fields with Gemini failed")
                return []

            # Convert to PDFField objects
            fields = []
            field_counter = 1

            # Handle both array and object responses
            if isinstance(fields_data, dict):
                fields_data = [fields_data]

            for field_data in fields_data:
                # Get or generate field name
                field_name = field_data.get("name")
                if not field_name and "label" in field_data:
                    # Create name from label if available
                    label = field_data.get("label", "")
                    field_name = label.strip().replace(" ", "_").replace(":", "").lower()
                    if not field_name:
                        field_name = f"field_{page_num}_{field_counter}"
                elif not field_name:
                    field_name = f"field_{page_num}_{field_counter}"

                field_counter += 1

                field = PDFField(
                    name=field_name,
                    page_num=page_num,
                    field_type=field_data.get("field_type", "UNKNOWN"),
                    rect=field_data.get("rect", [0, 0, 0, 0]),
                    label=field_data.get("label", ""),
                    label_rect=field_data.get("label_rect"),
                    confidence=field_data.get("confidence", 1.0)
                )

                fields.append(field)

            # Log summary
            logger.info(f"Extracted {len(fields)} fields with labels from page {page_num + 1}")
            return fields

        except Exception as e:
            logger.error(f"Error extracting fields and labels with Gemini: {e}")
            return []

    def extract_field_labels(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        """Extract all fields and their associated labels from the PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting fields and labels from: {pdf_path}")

        # Dictionary to store all fields
        all_fields = {}

        try:
            doc = fitz.open(pdf_path)

            # Process each page
            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1} of {len(doc)}")

                # Extract fields and labels using Gemini
                fields = self._extract_fields_and_labels(doc[page_num], page_num)

                # Add to results
                for field in fields:
                    all_fields[field.name] = field.to_dict()

                    # Log extracted field with label
                    logger.info(
                        f"Field: '{field.name}' (Page {field.page_num + 1}) [{field.field_type}] - Label: '{field.label}'")

            doc.close()
            return all_fields

        except Exception as e:
            logger.error(f"Failed to extract fields from {pdf_path}: {e}")
            raise

    def extract_labels_from_pymupdf_fields(self, pdf_path: str) -> Dict[str, str]:

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting labels for existing fields from: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)

            # First, extract basic fields using PyMuPDF
            pymupdf_fields = {}

            for page_num, page in enumerate(doc):
                try:
                    for widget in page.widgets():
                        if not widget.field_name:
                            continue

                        field_name = widget.field_name.strip()
                        pymupdf_fields[field_name] = {
                            "page_num": page_num,
                            "rect": [widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1],
                            "field_type": widget.field_type
                        }
                except Exception as e:
                    logger.error(f"Error extracting basic fields from page {page_num}: {e}")

            # Now use Gemini to identify labels for these fields
            field_labels = {}

            for page_num in range(len(doc)):
                page_fields = {k: v for k, v in pymupdf_fields.items() if v["page_num"] == page_num}

                if not page_fields:
                    continue

                # Get page image
                img = self._get_page_image(doc[page_num])

                # Initialize Gemini model
                model = genai.GenerativeModel(self.model_name)

                # Prepare data for the prompt
                fields_info = []
                for field_name, info in page_fields.items():
                    fields_info.append({
                        "name": field_name,
                        "rect": info["rect"],
                        "type": info["field_type"]
                    })

                # Create prompt focusing on identifying labels
                prompt = f"""
                I have a PDF form with the following form fields on this page:
                {json.dumps(fields_info, indent=2)}

                For each form field listed above, identify the associated label or descriptive text near the field.
                Consider proximity, alignment, and visual relationships to determine which text is meant to label each field.

                Return a JSON object where:
                - Each key is the original field name
                - Each value is the text of the label associated with that field

                Example:
                {{
                  "firstName": "First Name:",
                  "lastName": "Last Name:",
                  "dob": "Date of Birth"
                }}

                Return valid JSON only.
                """

                # Make request to Gemini
                for attempt in range(self.max_retries):
                    try:
                        response = model.generate_content([prompt, img])
                        response_text = response.text

                        # Try to extract JSON
                        try:
                            labels_data = json.loads(response_text)
                            # Update the field_labels dictionary
                            field_labels.update(labels_data)
                            break
                        except json.JSONDecodeError:
                            # Try to find JSON in the response
                            import re
                            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    labels_data = json.loads(json_match.group(1))
                                    field_labels.update(labels_data)
                                    break
                                except json.JSONDecodeError:
                                    pass

                            # One more attempt with any JSON-like structure
                            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    labels_data = json.loads(json_match.group(1))
                                    field_labels.update(labels_data)
                                    break
                                except json.JSONDecodeError:
                                    pass

                            logger.warning(f"Could not extract valid JSON for labels on attempt {attempt + 1}")

                    except Exception as e:
                        logger.warning(f"Gemini API error on labels attempt {attempt + 1}: {e}")

                    # Wait before retry
                    if attempt < self.max_retries - 1:
                        time.sleep(2 * (attempt + 1))

            # Log results
            for field_name, label in field_labels.items():
                logger.info(f"Field: '{field_name}' - Label: '{label}'")

            doc.close()
            return field_labels

        except Exception as e:
            logger.error(f"Failed to extract labels for existing fields: {e}")
            raise

    def extract_all_methods(self, pdf_path: str) -> Dict[str, Any]:
        """
        Comprehensive extraction using multiple methods.
        Returns combined results from PyMuPDF basic extraction and Gemini vision-based extraction.
        """
        logger.info(f"Performing comprehensive extraction on: {pdf_path}")

        results = {
            "ai_detected_fields": {},
            "pymupdf_fields": {},
            "field_labels": {}
        }

        try:
            # Method 1: Extract using Gemini vision approach
            try:
                ai_fields = self.extract_field_labels(pdf_path)
                results["ai_detected_fields"] = ai_fields
                logger.info(f"AI extraction found {len(ai_fields)} fields with labels")
            except Exception as e:
                logger.error(f"AI extraction failed: {e}")

            # Method 2: Extract labels for PyMuPDF fields
            try:
                field_labels = self.extract_labels_from_pymupdf_fields(pdf_path)
                results["field_labels"] = field_labels
                logger.info(f"Found labels for {len(field_labels)} PyMuPDF fields")
            except Exception as e:
                logger.error(f"PyMuPDF label extraction failed: {e}")

            # Method 3: Extract basic PyMuPDF fields
            try:
                doc = fitz.open(pdf_path)
                pymupdf_fields = {}

                for page_num, page in enumerate(doc):
                    for widget in page.widgets():
                        if not widget.field_name:
                            continue

                        field_name = widget.field_name.strip()

                        # Get the label from our extracted labels if available
                        label = results["field_labels"].get(field_name, "")

                        pymupdf_fields[field_name] = {
                            "page_num": page_num,
                            "field_type": widget.field_type,
                            "rect": [widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1],
                            "label": label
                        }

                        # Try to get field value
                        try:
                            if widget.field_type == 2:  # CHECKBOX
                                pymupdf_fields[field_name]["value"] = bool(widget.field_value)
                            else:
                                pymupdf_fields[field_name]["value"] = widget.field_value
                        except Exception:
                            pymupdf_fields[field_name]["value"] = None

                results["pymupdf_fields"] = pymupdf_fields
                logger.info(f"Basic PyMuPDF extraction found {len(pymupdf_fields)} fields")

                doc.close()
            except Exception as e:
                logger.error(f"Basic PyMuPDF extraction failed: {e}")

            return results

        except Exception as e:
            logger.error(f"Comprehensive extraction failed: {e}")
            return results


# Example usage
if __name__ == "__main__":
    # Initialize with your Gemini API key
    extractor = GeminiPDFLabelExtractor(
        gemini_api_key=API_KEY_3
    )

    # Option 1: Extract fields and labels using Gemini vision
    fields = extractor.extract_field_labels("D:\\demo\\Services\\WisconsinLLC.pdf")
    print(f"Extracted {len(fields)} fields with labels")

    # Option 2: Extract labels for existing PyMuPDF fields
    labels = extractor.extract_labels_from_pymupdf_fields("D:\\demo\\Services\\WisconsinLLC.pdf")
    print(f"Found labels for {len(labels)} PyMuPDF fields")

    # Option 3: Comprehensive extraction using all methods
    results = extractor.extract_all_methods("D:\\demo\\Services\\WisconsinLLC.pdf")
    print(f"Comprehensive extraction results:")
    print(f"- AI detected fields: {len(results['ai_detected_fields'])}")
    print(f"- PyMuPDF fields: {len(results['pymupdf_fields'])}")
    print(f"- Field labels: {len(results['field_labels'])}")