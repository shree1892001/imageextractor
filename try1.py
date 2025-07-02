import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
import json
import os
from Common.constants import *


class PDFFieldExtractor:
    def __init__(self, json_path=None):
        """
        Initialize PDF field extractor with JSON data source

        Args:
            json_path (str, optional): Path to JSON file with form fields
        """
        self.fillable_fields = {}
        self.original_data = {}

        # Load fields from JSON if path is provided
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    self.original_data = json.load(f)
                    self.fillable_fields = self._flatten_json(self.original_data)
            except Exception as e:
                print(f"Error loading JSON file: {e}")

    def _flatten_json(self, data, parent_key='', sep='_'):
        """
        Flatten nested JSON dictionary

        Args:
            data (dict): Nested dictionary to flatten
            parent_key (str): Parent key for nested items
            sep (str): Separator for nested keys

        Returns:
            dict: Flattened dictionary
        """
        items = []

        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                # Recursively flatten nested dictionaries or lists
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten_json(v, new_key, sep=sep).items())
                else:
                    # For simple values, add to items
                    items.append((new_key, v))

        elif isinstance(data, list):
            # Handle list of dictionaries
            for i, item in enumerate(data):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)

                # Recursively flatten list items
                if isinstance(item, (dict, list)):
                    items.extend(self._flatten_json(item, new_key, sep=sep).items())
                else:
                    # For simple list values
                    items.append((new_key, item))
        else:
            # If it's a simple value, return as is
            items.append((parent_key, data))

        return dict(items)

    def _find_best_field_match(self, pdf_line, fillable_fields):
        """
        Find the best matching field for a given PDF line

        Args:
            pdf_line (str): Line from PDF
            fillable_fields (dict): Flattened JSON fields

        Returns:
            tuple: (best matching key, value) or (None, None)
        """
        best_match = None
        best_score = 0
        best_value = None

        for key, value in fillable_fields.items():
            # Convert both to lowercase for case-insensitive matching
            key_lower = key.lower().replace('_', ' ')
            line_lower = pdf_line.lower()

            # Calculate matching score
            if key_lower in line_lower:
                # More precise scoring
                score = len(key_lower) / len(line_lower)

                if score > best_score:
                    best_match = key
                    best_score = score
                    best_value = value

        return best_match, best_value

    def extract_text_from_images(self, input_pdf_path):
        """Converts PDF pages to images and extracts text using OCR."""
        images = convert_from_path(input_pdf_path)
        pdf_text_data = {}

        for page_num, img in enumerate(images, start=1):
            # Convert image to OpenCV format
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for better OCR

            # Extract text using Tesseract OCR
            text = pytesseract.image_to_string(img_cv)

            # Store text per page
            pdf_text_data[page_num] = text.split("\n")  # Store as lines

        return pdf_text_data

    def fill_pdf_fields(self, input_pdf_path, output_pdf_path):
        """
        Fill PDF fields using flattened JSON data

        Args:
            input_pdf_path (str): Path to input PDF
            output_pdf_path (str): Path to save annotated PDF
        """
        # Extract text from PDF images
        pdf_text_data = self.extract_text_from_images(input_pdf_path)

        # Prepare debugging logs
        debug_mapping = {}

        # Open PDF for annotation
        doc = fitz.open(input_pdf_path)

        for page_num, lines in pdf_text_data.items():
            used_rects = set()  # Track used placeholder positions

            for line in lines:
                # Find best matching field for this line
                best_match, best_value = self._find_best_field_match(line, self.fillable_fields)

                if best_match:
                    # Locate placeholder
                    matches = doc[page_num - 1].search_for("___________________")

                    if matches:
                        for rect in matches:
                            if rect in used_rects:
                                continue  # Skip if already used

                            x0, y0, x1, y1 = rect

                            # Erase the placeholder
                            doc[page_num - 1].insert_text((x0, y0), " " * 15, fontsize=10, color=(1, 1, 1))

                            # Insert the value inside the text box
                            doc[page_num - 1].insert_text((x0 + 5, y0 + 8), str(best_value), fontsize=10,
                                                          color=(0, 0, 0))

                            used_rects.add(rect)  # Mark this placeholder as used

                            # Log the mapping for debugging
                            debug_mapping[line] = {
                                'matched_key': best_match,
                                'value': best_value
                            }
                            break

        doc.save(output_pdf_path)
        print(f"Annotated PDF saved as: {output_pdf_path}")

        # Print detailed debugging information
        print("\nDetailed Field Mapping:")
        print(json.dumps(debug_mapping, indent=2))
        print("\nAll Flattened Fields:")
        print(json.dumps(self.fillable_fields, indent=2))


# Usage Example
def main():
    input_pdf = "D:\\demo\\Services\\Maine.pdf"
    output_pdf = "D:\\demo\\Services\\fill_smart13.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"

    # Create extractor with JSON data
    extractor = PDFFieldExtractor(json_path=json_path)
    extractor.fill_pdf_fields(input_pdf, output_pdf)


if __name__ == "__main__":
    main()