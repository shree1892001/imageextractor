import fitz
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import cv2
import pytesseract
from difflib import SequenceMatcher

@dataclass
class OCRElement:
    text: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    confidence: float = 0.0

@dataclass
class FormField:
    name: str
    value: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    page_num: Optional[int] = None

class TessoFiller:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.7
        self.position_weight = 0.3
        self.text_weight = 0.7

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return morph

    def extract_ocr_elements(self, pdf_path: str) -> Dict[int, List[OCRElement]]:
        """Extract OCR elements with improved accuracy using both PyMuPDF and Tesseract."""
        doc = fitz.open(pdf_path)
        page_elements: Dict[int, List[OCRElement]] = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Increase resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Preprocess image for better OCR
            processed_img = self.preprocess_image(img_np)
            
            # Get OCR data from Tesseract with confidence scores
            ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            # Get words from PyMuPDF for accurate positioning
            words = page.get_text("words")
            
            if page_num not in page_elements:
                page_elements[page_num] = []

            # Combine results from both OCR engines
            for i, word in enumerate(words):
                # Find corresponding Tesseract result
                tesseract_match = self.find_matching_tesseract_word(word[4], ocr_data)
                
                element = OCRElement(
                    text=word[4],
                    bbox=(word[0], word[1], word[2], word[3]),
                    page_num=page_num,
                    confidence=tesseract_match['conf'] if tesseract_match else 0.0
                )
                page_elements[page_num].append(element)

        return page_elements

    def find_matching_tesseract_word(self, word: str, ocr_data: Dict) -> Optional[Dict]:
        best_ratio = 0
        best_match = None
        
        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue
            
            ratio = SequenceMatcher(None, word.lower(), text.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = {
                    'text': text,
                    'conf': float(ocr_data['conf'][i])
                }
        
        return best_match if best_ratio > 0.8 else None

    def extract_keywords(self, text: str) -> List[str]:
        # Split into words and normalize
        words = text.lower().split()
        # Remove common stop words and keep meaningful terms
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
        keywords = [word for word in words if word not in stop_words]
        return keywords

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        # Preprocess and extract keywords for each text
        processed_texts = []
        for text in texts:
            keywords = self.extract_keywords(text)
            processed_text = ' '.join(keywords) if keywords else text
            processed_texts.append(processed_text)
        return self.model.encode(processed_texts)

    def match_and_fill_form(self, pdf_path: str, form_data: Dict[str, str], output_path: str):
        """Match and fill form fields with improved accuracy using sequential page-by-page processing."""
        try:
            doc = fitz.open(pdf_path)
            remaining_fields = form_data.copy()
            all_page_elements = {}
            
            # First pass: Extract all page elements and their positions
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                processed_img = self.preprocess_image(img_np)
                
                ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                words = page.get_text("words")
                
                page_elements = []
                for word in words:
                    tesseract_match = self.find_matching_tesseract_word(word[4], ocr_data)
                    element = OCRElement(
                        text=word[4],
                        bbox=(word[0], word[1], word[2], word[3]),
                        page_num=page_num,
                        confidence=tesseract_match['conf'] if tesseract_match else 0.0
                    )
                    page_elements.append(element)
                
                all_page_elements[page_num] = page_elements
                
                # Clear memory
                del processed_img
                del img_np
                del img
                del pix
            
            # Second pass: Process fields sequentially page by page
            for page_num in range(len(doc)):
                print(f"Processing page {page_num + 1}...")
                page = doc[page_num]
                page_elements = all_page_elements[page_num]
                
                if not page_elements:
                    continue
                
                # Sort page elements by vertical position for sequential processing
                page_elements.sort(key=lambda x: (x.bbox[1], x.bbox[0]))
                
                texts = [elem.text for elem in page_elements]
                text_embeddings = self.compute_embeddings(texts)
                
                # Process fields in order of appearance on the page
                filled_positions = []
                for element in page_elements:
                    best_field = None
                    best_score = 0
                    
                    for field_name, field_value in remaining_fields.items():
                        # Extract keywords and calculate semantic similarity
                        field_keywords = self.extract_keywords(field_name)
                        element_keywords = self.extract_keywords(element.text)
                        
                        # Compute embeddings for both sets of keywords
                        field_embedding = self.model.encode([' '.join(field_keywords)] if field_keywords else [field_name])
                        element_embedding = self.model.encode([' '.join(element_keywords)] if element_keywords else [element.text])
                        
                        # Calculate semantic similarity
                        text_similarity = cosine_similarity(field_embedding, element_embedding)[0][0]
                        
                        # Boost similarity if keywords overlap
                        keyword_overlap = len(set(field_keywords) & set(element_keywords))
                        if keyword_overlap > 0:
                            text_similarity = text_similarity * (1 + 0.1 * keyword_overlap)
                        
                        # Calculate position score based on vertical ordering
                        position_score = 1.0
                        if filled_positions:
                            # Prefer positions below the last filled field
                            last_y = filled_positions[-1][3]  # bottom of last filled field
                            if element.bbox[1] < last_y:  # if current element is above last filled
                                position_score = 0.5
                        
                        final_score = (self.text_weight * text_similarity + 
                                      self.position_weight * position_score)
                        
                        if final_score > best_score and final_score >= self.similarity_threshold:
                            best_score = final_score
                            best_field = (field_name, field_value)
                    
                    if best_field:
                        try:
                            field_name, field_value = best_field
                            # Calculate insertion point below the matched element
                            insert_x = element.bbox[0]
                            insert_y = element.bbox[3] + 2
                            
                            # Avoid overlapping with previously filled fields
                            while any(abs(pos[3] - insert_y) < 12 for pos in filled_positions):
                                insert_y += 12
                            
                            page.insert_text(
                                point=(insert_x, insert_y),
                                text=field_value,
                                fontsize=11,
                                color=(0, 0, 0)
                            )
                            
                            filled_positions.append(element.bbox)
                            remaining_fields.pop(field_name)
                            print(f"Filled field '{field_name}' on page {page_num + 1}")
                        except Exception as e:
                            print(f"Error filling field '{field_name}' on page {page_num + 1}: {str(e)}")
                
                # Save progress after each page
                doc.save(output_path)
            
            # Report unfilled fields
            if remaining_fields:
                print("\nWarning: The following fields could not be filled:")
                for field_name in remaining_fields:
                    print(f"- {field_name}")
            
            doc.close()
            print(f"\nForm filling completed. Output saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

def main():
    # Example usage
    pdf_path = "input.pdf"
    output_path = "output.pdf"
    form_data = {
        "Name": "John Doe",
        "Date": "2024-01-20",
        "Address": "123 Main St"
    }
    
    filler = TessoFiller()
    filler.match_and_fill_form(pdf_path, form_data, output_path)

if __name__ == "__main__":
    main()