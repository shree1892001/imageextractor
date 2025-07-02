import os
import fitz
import docx
from abc import ABC, abstractmethod

class TextExtractor(ABC):


    @abstractmethod
    async def extract_text(self, file_path: str) -> str:

        pass

class PDFTextExtractor(TextExtractor):


    async def extract_text(self, file_path: str) -> str:

        extracted_text = []
        try:
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text = page.get_text("text")
                    extracted_text.append(text)
            return "\n".join(extracted_text).strip()
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

class DOCXTextExtractor(TextExtractor):


    async def extract_text(self, file_path: str) -> str:

        extracted_text = []
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                extracted_text.append(para.text)
            return "\n".join(extracted_text).strip()
        except Exception as e:
            return f"Error extracting text from DOCX: {str(e)}"

class ExtractorFactory:


    @staticmethod
    def get_extractor(file_path: str) -> TextExtractor:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[-1].lower()

        if file_extension == ".pdf":
            return PDFTextExtractor()
        elif file_extension == ".docx":
            return DOCXTextExtractor()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")