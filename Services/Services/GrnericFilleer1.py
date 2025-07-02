import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from PyPDF2 import PdfReader
from fillpdf import fillpdfs
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *


class PDFField(BaseModel):
    name: str
    type: str
    editable: bool
    options: Optional[List[str]] = None
    required: bool = False


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str
    confidence: float
    suggested_value: Any
    field_type: str
    editable: bool
    reasoning: str

    @field_validator('confidence')
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)


class ProcessingResult(BaseModel):
    status: str
    matches: List[FieldMatch]
    unmatched_fields: List[str]
    success_rate: float
    execution_time: float
    non_editable_fields: List[str]





class IntelligentFormFiller:
    def __init__(self, api_key: str):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt="You are an expert at intelligent form field matching, focusing on semantic understanding and context."
        )
        self.processing_history = []

    def extract_pdf_fields(self, pdf_path: str) -> List[PDFField]:
        fields = []
        reader = PdfReader(pdf_path)

        for page in reader.pages:
            if '/Annots' in page:
                for annot in page['/Annots']:
                    obj = annot.get_object()
                    if obj['/Subtype'] == '/Widget':
                        field = self._process_pdf_field(obj)
                        if field:
                            fields.append(field)
        return fields

    def _process_pdf_field(self, field_obj: Dict) -> Optional[PDFField]:
        try:
            field_name = field_obj.get('/T', '')
            if not field_name:
                return None

            field_type = field_obj.get('/FT', '/Tx')
            mapped_type = {
                '/Tx': 'text',
                '/Btn': 'checkbox',
                '/Ch': 'select',
                '/Sig': 'signature'
            }.get(field_type, 'text')

            flags = field_obj.get('/Ff', 0)
            editable = not bool(flags & 1)
            required = bool(flags & 2)

            options = None
            if mapped_type in ('select', 'radio'):
                opt = field_obj.get('/Opt', [])
                options = [o if isinstance(o, str) else o[1] for o in opt]

            return PDFField(
                name=field_name,
                type=mapped_type,
                editable=editable,
                options=options,
                required=required
            )
        except Exception as e:
            print(f"Error processing field: {str(e)}")
            return None

    def flatten_json(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        items = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                items.update(self.flatten_json(value, new_key))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(self.flatten_json(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = value
        return items

    async def match_fields(
            self,
            json_data: Dict[str, Any],
            pdf_fields: List[PDFField]
    ) -> Dict[str, Any]:
        flat_json = self.flatten_json(json_data)

        pdf_field_info = [{
            'name': f.name,
            'type': f.type,
            'editable': f.editable,
            'options': f.options,
            'required': f.required
        } for f in pdf_fields]

        prompt = FIELD_MATCHING_PROMPT1.format(
            json_data=json.dumps(flat_json, indent=2),
            pdf_fields=json.dumps(pdf_field_info, indent=2)
        )

        try:
            response = await self.agent.run(prompt)
            result = json.loads(response.data.strip().replace("```json", "").replace("```", "").strip())
            return self._validate_matches(result, pdf_fields)
        except Exception as e:
            print(f"Error in field matching: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {"matches": []}

    def _validate_matches(self, result: Dict[str, Any], pdf_fields: List[PDFField]) -> Dict[str, Any]:
        pdf_field_dict = {f.name: f for f in pdf_fields}
        valid_matches = []

        for match in result.get("matches", []):
            try:
                if match.get("pdf_field") in pdf_field_dict:
                    pdf_field = pdf_field_dict[match["pdf_field"]]
                    match["suggested_value"] = self.preprocess_value(
                        match["suggested_value"],
                        match["field_type"]
                    )
                    valid_matches.append(match)
            except Exception as e:
                print(f"Error validating match: {str(e)}")
                continue

        result["matches"] = valid_matches
        return result

    def preprocess_value(self, value: Any, field_type: str) -> str:
        if value is None:
            return ""

        if field_type == 'checkbox':
            if isinstance(value, bool):
                return "Yes" if str(value).lower() in ('true', 'yes', '1', 'on') else "Off"

        if isinstance(value, (int, float)):
            return str(value)

        return str(value).strip()

    async def process_form(
            self,
            template_pdf: str,
            json_data: Dict[str, Any],
            output_pdf: str
    ) -> ProcessingResult:
        start_time = datetime.now()

        try:
            print(f"\n📄 Processing PDF: {template_pdf}")
            pdf_fields = self.extract_pdf_fields(template_pdf)
            print(f"Found {len(pdf_fields)} PDF fields")

            print("\n🤖 Matching fields with AI...")
            result = await self.match_fields(json_data, pdf_fields)

            valid_matches = [FieldMatch(**match) for match in result["matches"]]

            fill_data = {}
            non_editable = []
            for match in valid_matches:
                if match.editable:
                    fill_data[match.pdf_field] = match.suggested_value
                else:
                    non_editable.append(match.pdf_field)

            print(f"\n📝 Filling {len(fill_data)} editable fields")
            if fill_data:
                fillpdfs.write_fillable_pdf(template_pdf, output_pdf, fill_data)

            flat_json = self.flatten_json(json_data)
            matched_fields = {match.json_field for match in valid_matches}
            unmatched = [f for f in flat_json.keys() if f not in matched_fields]
            success_rate = (len(valid_matches) / len(flat_json)) * 100 if flat_json else 0

            result = ProcessingResult(
                status="success",
                matches=valid_matches,
                unmatched_fields=unmatched,
                success_rate=success_rate,
                execution_time=(datetime.now() - start_time).total_seconds(),
                non_editable_fields=non_editable
            )

            self.processing_history.append(result)
            return result

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return ProcessingResult(
                status="error",
                matches=[],
                unmatched_fields=list(self.flatten_json(json_data).keys()),
                success_rate=0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                non_editable_fields=[]
            )

    async def run(self, template_pdf: str, json_path: str, output_pdf: str):
        try:
            print("\n🚀 Starting Intelligent Form Processing...")

            if not os.path.exists(template_pdf):
                raise FileNotFoundError(f"Template PDF not found: {template_pdf}")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            with open(json_path, 'r') as f:
                json_data = json.load(f)

            result = await self.process_form(template_pdf, json_data, output_pdf)

            print(f"\n📊 Processing Results:")
            print(f"Status: {result.status}")
            print(f"Success Rate: {result.success_rate:.1f}%")
            print(f"Execution Time: {result.execution_time:.2f} seconds")

            if result.matches:
                print("\n✅ Successful Matches:")
                for match in result.matches:
                    print(f"\n• JSON Field: {match.json_field}")
                    print(f"  PDF Field: {match.pdf_field}")
                    print(f"  Type: {match.field_type}")
                    print(f"  Editable: {'Yes' if match.editable else 'No'}")
                    print(f"  Confidence: {match.confidence:.2f}")
                    print(f"  Value: {match.suggested_value}")
                    print(f"  Reasoning: {match.reasoning}")

            if result.non_editable_fields:
                print("\n🔒 Non-Editable Fields:")
                for field in result.non_editable_fields:
                    print(f"• {field}")

            if result.unmatched_fields:
                print("\n⚠️ Unmatched Fields:")
                for field in result.unmatched_fields:
                    print(f"• {field}")

            print(f"\n💾 Filled PDF saved to: {output_pdf}")
            return result

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            print(traceback.format_exc())


async def main():
    form_filler = IntelligentFormFiller(API_KEY)

    template_pdf = "D:\\demo\\Services\\PennsylvaniaLLC.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
    output_pdf = "D:\\demo\\Services\\filled_form11.pdf"

    try:
        result = await form_filler.run(template_pdf, json_path, output_pdf)

        print(f"\nTotal forms processed: {len(form_filler.processing_history)}")

        return result
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None


if __name__ == "__main__":
    asyncio.run(main())