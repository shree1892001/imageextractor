import asyncio
import json
import os
import re
import shutil
from typing import Dict, Any, List, Optional
from Common.constants import *
import fitz
import numpy as np
import cv2
import pytesseract
from pypdf import PdfReader, PdfWriter
from jsonschema import validate, ValidationError
from difflib import SequenceMatcher

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel, field_validator
from pydantic_ai.providers.google_gla import GoogleGLAProvider

# Set environment variables for determinism
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
np.random.seed(42)

API_KEYS = {
    "field_matcher": API_KEY_1,
}

FIELD_MATCHING_RULES = """
## STRICT MATCHING RULES
1. Use EXACT PDF field names from provided list
2. Follow example value formats precisely
3. Priority: Direct matches > Contextual matches > Leave blank
4. Confidence: 1.0=Exact, 0.9=Direct, 0.7=Contextual
5. Never guess values - empty > incorrect
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["json_field", "pdf_field", "confidence", "suggested_value", "reasoning"],
                "properties": {
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "pdf_field": {"type": "string"},
                    "json_field": {"type": "string"},
                    "suggested_value": {},
                    "reasoning": {"type": "string"}
                }
            }
        }
    },
    "required": ["matches"]
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str
    confidence: float
    suggested_value: Any
    reasoning: str

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be 0-1")
        return round(float(v), 2)


class OCRFieldMatch(FieldMatch):
    ocr_text: str
    page_num: int
    x1: float
    y1: float
    x2: float
    y2: float


class MultiAgentFormFiller:
    def __init__(self):
        model_settings = ModelSettings(
            temperature=0.3,
            top_p=0.7,
            max_tokens=2048,
            seed=42
        )

        self.agent = Agent(
            model=GeminiModel(
                model_name="gemini-1.5-flash",
                provider=GoogleGLAProvider(
                    api_key=API_KEYS["field_matcher"]
                ),
                settings=model_settings
            ),
            system_prompt=f"""PDF field mapping expert. Rules:\n{FIELD_MATCHING_RULES}"""
        )
        self.previous_results = []

    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        fields = {}
        for page_num, page in enumerate(doc):
            for widget in page.widgets():
                if widget.field_name:
                    info = {
                        "page_num": page_num,
                        "type": widget.field_type,
                        "rect": [widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1],
                        "flags": widget.field_flags,
                        "is_readonly": bool(widget.field_flags & 1),
                        "current_value": widget.field_value
                    }
                    fields[widget.field_name.strip()] = info
        doc.close()
        return fields

    async def extract_ocr_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        ocr_results = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            data = pytesseract.image_to_data(
                binary,
                config=r'--oem 1 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            for i in range(len(data['text'])):
                if data['conf'][i] > 60 and data['text'][i].strip():
                    ocr_results.append({
                        "page_num": page_num,
                        "text": data['text'][i].strip(),
                        "confidence": data['conf'][i] / 100,
                        "position": {
                            "x1": data['left'][i],
                            "y1": data['top'][i],
                            "x2": data['left'][i] + data['width'][i],
                            "y2": data['top'][i] + data['height'][i]
                        }
                    })
        doc.close()
        return ocr_results

    async def match_and_fill_fields(self, pdf_path: str, json_data: Dict[str, Any],
                                    output_pdf: str, max_retries: int = 3) -> bool:
        backup_pdf = f"{pdf_path}.backup"
        shutil.copy2(pdf_path, backup_pdf)

        pdf_fields = await self.extract_pdf_fields(pdf_path)
        ocr_text = await self.extract_ocr_text(pdf_path)
        flat_json = self.flatten_json(json_data)

        prompt = self._build_prompt(flat_json, pdf_fields, ocr_text)
        best_result = None

        for attempt in range(max_retries):
            try:
                response = await self.agent.run(prompt)
                result = self.parse_ai_response(response.data)
                if result and self._is_more_consistent(result):
                    best_result = result
                    self.previous_results.append(result)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

        if not best_result:
            return False

        temp_pdf = f"{output_pdf}.temp"
        shutil.copy2(pdf_path, temp_pdf)

        try:
            ordered_matches = sorted(
                best_result["matches"],
                key=lambda m: (m.pdf_field, pdf_fields[m.pdf_field]["page_num"])
            )
            self._fill_pdf(temp_pdf, ordered_matches, pdf_fields)
            self._finalize_pdf(temp_pdf, output_pdf)
            return self._verify_filled(output_pdf)
        finally:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    def _build_prompt(self, flat_json: Dict, pdf_fields: Dict, ocr_text: List) -> str:
        return f"""Match JSON data to PDF fields using these rules:
{FIELD_MATCHING_RULES}

JSON Data:
{json.dumps(flat_json, indent=2, cls=NumpyEncoder)}

PDF Fields:
{json.dumps(list(pdf_fields.keys()), indent=2)}

OCR Elements (for context):
{json.dumps(ocr_text[:50], indent=2)}

Respond ONLY with valid JSON matching this schema:
{json.dumps(RESPONSE_SCHEMA, indent=2)}"""

    def parse_ai_response(self, response_text: str) -> Optional[Dict]:
        try:
            cleaned = re.sub(r'[\s\S]*?({[\s\S]*})[\s\S]*', r'\1', response_text)
            data = json.loads(cleaned)
            validate(instance=data, schema=RESPONSE_SCHEMA)
            return {
                "matches": [
                    FieldMatch(**m) for m in data["matches"]
                ]
            }
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Invalid response: {e}")
            return None

    def _is_more_consistent(self, result: Dict) -> bool:
        if not self.previous_results:
            return True
        current = {(m.pdf_field, m.json_field) for m in result["matches"]}
        matches = sum(
            1 for prev in self.previous_results
            if len(current & {(m.pdf_field, m.json_field) for m in prev["matches"]}) / len(current) > 0.8
        )
        return matches / len(self.previous_results) > 0.5

    def _fill_pdf(self, pdf_path: str, matches: List[FieldMatch], fields: Dict) -> None:
        doc = fitz.open(pdf_path)
        for match in matches:
            if match.pdf_field not in fields:
                continue
            info = fields[match.pdf_field]
            page = doc[info["page_num"]]
            for widget in page.widgets():
                if widget.field_name == match.pdf_field and not info["is_readonly"]:
                    widget.field_value = str(match.suggested_value)
                    widget.update()
        doc.saveIncremental()

    def _finalize_pdf(self, src: str, dest: str) -> None:
        reader = PdfReader(src)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        with open(dest, "wb") as f:
            writer.write(f)

    def _verify_filled(self, pdf_path: str) -> bool:
        reader = PdfReader(pdf_path)
        return len([f for f in reader.get_fields() if f.get("/V")]) > 0

    def flatten_json(self, data: Dict, prefix: str = "") -> Dict:
        items = {}
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self.flatten_json(v, key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.update(self.flatten_json({str(i): item}, key))
            else:
                items[key] = v
        return items


async def main():
    form_filler = MultiAgentFormFiller()
    success = await form_filler.match_and_fill_fields(
        "input.pdf",
        {"entity": {"name": "Test LLC", "address": "123 Main St"}},
        "output.pdf"
    )
    print(f"Success: {success}")


if __name__ == "__main__":
    asyncio.run(main())