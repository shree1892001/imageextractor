import fitz  # PyMuPDF
import tempfile
from typing import List, Dict, Any, Optional
import os
import json
import re
from dataclasses import dataclass
from Common.constants import *
from psycopg2 import sql
from Extractor.ImageExtractor import ImageTextExtractor
from Extractor.Paddle import flatten_json
from Factories.DocumentFactory import DocumentExtractorFactory, DocumentExtractor
from utils.ApplicationConnection import ApplicationConnection

@dataclass
class DocumentInfo:
    document_type: str
    confidence_score: float
    extracted_data: Dict[str, Any]

class DynamicDatabaseHandler:
    def __init__(self, app_connection):
        self.connection = app_connection.connect()
        self.connection.autocommit = False
        self.cursor = self.connection.cursor()

    def create_or_alter_table(self, table_name, json_data):
        table_name = table_name.translate(str.maketrans("", "", "'"))

        try:
            self.cursor.execute(sql.SQL("SELECT to_regclass(%s)"), [table_name])
            table_exists = self.cursor.fetchone()[0] is not None

            if not table_exists:
                columns = ", ".join([f"\"{key}\" TEXT" for key in json_data.keys()])
                create_table_query = sql.SQL(f"CREATE TABLE {table_name} ({columns})")
                self.cursor.execute(create_table_query)
                self.connection.commit()
            else:
                self.cursor.execute(
                    sql.SQL("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                    """), [table_name]
                )
                existing_columns = {row[0] for row in self.cursor.fetchall()}

                for key in json_data.keys():
                    if key not in existing_columns:
                        try:
                            alter_table_query = sql.SQL(
                                f'ALTER TABLE {table_name} ADD COLUMN "{key}" TEXT'
                            )
                            self.cursor.execute(alter_table_query)
                            self.connection.commit()
                        except Exception as e:
                            self.connection.rollback()
                            print(f"Warning: Could not add column {key}: {str(e)}")

        except Exception as e:
            self.connection.rollback()
            raise e

    def insert_data(self, table_name, json_data):
        try:
            for key, value in json_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    json_data[key] = json.dumps(value)

            keys = ", ".join([f'"{key}"' for key in json_data.keys()])
            placeholders = ", ".join([f"%({key})s" for key in json_data.keys()])
            insert_query = sql.SQL(f"INSERT INTO {table_name} ({keys}) VALUES ({placeholders})")
            self.cursor.execute(insert_query, json_data)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e

    def close(self):
        try:
            self.connection.commit()
        except Exception:
            self.connection.rollback()
        finally:
            self.cursor.close()
            self.connection.close()

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
            raise ValueError(f"Failed to classify document: {str(e)}")

class DocumentProcessor:
    def __init__(self, api_key: str, db_connection_params: dict):
        self.api_key = api_key
        self.db_handler = DynamicDatabaseHandler(ApplicationConnection(**db_connection_params))
        self.classifier = DocumentClassifier(api_key)

    def process_file(self, file_path: str, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Process either a single image or a PDF with multiple images"""
        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path, min_confidence)
        else:
            result = self._process_single_image(file_path, min_confidence)
            return [result] if result else []

    def _process_pdf(self, pdf_path: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Extract and process images from a PDF"""
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

        # Process multi-page PDF or single-page PDF with multiple images
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
                print(f"Low confidence ({doc_info.confidence_score}) in document classification for {image_path}")
                return None

            extractor = DocumentExtractorFactory.get_extractor(doc_info.document_type, self.api_key)
            extracted_data = extractor.extract_fields(image_path)

            if extractor.validate_fields(extracted_data):
                flattened_data = flatten_json(extracted_data)
                table_name = f"extracted_data_{doc_info.document_type.lower()}"
                self.db_handler.create_or_alter_table(table_name, flattened_data)
                self.db_handler.insert_data(table_name, flattened_data)

                return {
                    "document_type": doc_info.document_type,
                    "confidence_score": doc_info.confidence_score,
                    "extracted_data": extracted_data,
                    "status": "success"
                }
            else:
                print(f"Validation failed for {image_path}")
                return None

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

if __name__ == "__main__":
    config = {
        "api_key": API_KEY,
        "db_connection_params": {
            "host": DB_HOST,
            "dbname": DB_NAME,
            "user": DB_USER,
            "password": DB_PASS
        }
    }

    processor = DocumentProcessor(
        config["api_key"],
        config["db_connection_params"]
    )

    results = processor.process_file("D:\\TextExtractor\\Extractor\\images\\pancard.jpg")

    for doc in results:
        print(f"Document Type: {doc['document_type']}")
        print(f"Confidence Score: {doc['confidence_score']}")
        print(f"Extracted Data: {doc['extracted_data']}")