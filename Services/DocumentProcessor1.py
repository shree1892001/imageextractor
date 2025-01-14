import fitz
import tempfile
from typing import List, Dict, Any, Optional
import os
import json
from Common.constants import *
from dataclasses import dataclass
from Extractor.ImageExtractor import ImageTextExtractor
from Extractor.Paddle import flatten_json
from Factories.DocumentFactory import DocumentExtractorFactory, DocumentExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentInfo:
    document_type: str
    confidence_score: float
    extracted_data: Dict[str, Any]

class DocumentClassifier:
    def __init__(self, api_key: str):
        self.image_extractor = ImageTextExtractor(api_key)

    def identify_document_type(self, image_path: str) -> DocumentInfo:
        classification_prompt = CLASSIFICATION_PROMPT
        try:
            response = self.image_extractor.query_gemini_llm(image_path, classification_prompt)
            clean_json = json.loads(response.strip().replace("```json", "").replace("```", "").strip())

            return DocumentInfo(
                document_type=clean_json["document_type"],
                confidence_score=float(clean_json["confidence_score"]),
                extracted_data=clean_json
            )
        except Exception as e:
            logger.exception("Failed to classify document")
            raise ValueError(f"Failed to classify document: {str(e)}")

class DocumentProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.classifier = DocumentClassifier(api_key)

    def process_file(self, file_path: str, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path, min_confidence)
        else:
            result = self._process_single_image(file_path, min_confidence)
            return [result] if result else []

    def _process_pdf(self, pdf_path: str, min_confidence: float) -> List[Dict[str, Any]]:
        results = []
        pdf_document = fitz.open(pdf_path)

        if pdf_document.page_count == 1:
            page = pdf_document[0]
            images = page.get_images(full=True)
            if len(images) == 1:
                with tempfile.TemporaryDirectory() as temp_dir:
                    img_data = pdf_document.extract_image(images[0][0])
                    temp_image_path = os.path.join(temp_dir, f"single_page_image.png")

                    with open(temp_image_path, "wb") as img_file:
                        img_file.write(img_data["image"])

                    result = self._process_single_image(temp_image_path, min_confidence)
                    if result:
                        results.append(result)
                return results

        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                images = page.get_images(full=True)

                for img_index, img_info in enumerate(images):
                    img_data = pdf_document.extract_image(img_info[0])
                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}_img_{img_index}.png")

                    with open(temp_image_path, "wb") as img_file:
                        img_file.write(img_data["image"])

                    result = self._process_single_image(temp_image_path, min_confidence)
                    if result:
                        results.append(result)

        return results

    def _process_single_image(self, image_path: str, min_confidence: float) -> Optional[Dict[str, Any]]:
        """Process a single image with document type detection"""
        try:
            doc_info = self.classifier.identify_document_type(image_path)

            if doc_info.confidence_score < min_confidence:
                logger.warning(f"Low confidence ({doc_info.confidence_score}) in document classification for {image_path}")
                return None

            extractor = DocumentExtractorFactory.get_extractor(doc_info.document_type, self.api_key)
            extracted_data = extractor.extract_fields(image_path)

            if extractor.validate_fields(extracted_data):
                flattened_data = flatten_json(extracted_data)

                return {
                    "document_type": doc_info.document_type,
                    "confidence_score": doc_info.confidence_score,
                    "extracted_data": flattened_data,
                    "status": "success"
                }
            else:
                logger.warning(f"Validation failed for {image_path}")
                return None

        except Exception as e:
            logger.exception(f"Error processing image {image_path}")
            return None