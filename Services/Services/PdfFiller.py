from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
from PyPDF2 import PdfReader
from fillpdf import fillpdfs
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import traceback
from Common.constants import *


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str
    confidence: float
    suggested_value: Any
    reasoning: str

    @field_validator('pdf_field')
    def validate_pdf_field(cls, v):
        if v is None:
            raise ValueError("PDF field cannot be None")
        return str(v)

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


def flatten_json(data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested JSON structure into dot notation."""
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            items.update(flatten_json(value, new_key))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_json(item, f"{new_key}[{i}]"))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = value
    return items


def unflatten_json(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flattened dictionary with dot notation back to nested dictionary."""
    result = {}
    for key, value in flat_dict.items():
        parts = key.split('.')
        target = result
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                list_key = part[:part.index('[')]
                index = int(part[part.index('[') + 1:part.index(']')])
                if list_key not in target:
                    target[list_key] = []
                while len(target[list_key]) <= index:
                    target[list_key].append({})
                target = target[list_key][index]
            else:
                target = target.setdefault(part, {})
        last_part = parts[-1]
        if '[' in last_part and ']' in last_part:
            list_key = last_part[:last_part.index('[')]
            index = int(last_part[last_part.index('[') + 1:last_part.index(']')])
            if list_key not in target:
                target[list_key] = []
            while len(target[list_key]) <= index:
                target[list_key].append(None)
            target[list_key][index] = value
        else:
            target[last_part] = value
    return result


SYSTEM_PROMPT = FILER_PROMPT

class GeminiFormFiller:
    def __init__(self, api_key: str):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt=SYSTEM_PROMPT,
        )
        self.processing_history = []

    def preprocess_value(self, value: Any) -> Any:
        """Preprocess values to ensure they're in a format suitable for PDF fields"""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            # Remove any null strings
            if value.lower() == "null":
                return ""
            return value.strip()
        return str(value)

    async def match_and_format_fields(
            self,
            json_data: Dict[str, Any],
            pdf_fields: List[str]
    ) -> Dict[str, Any]:
        """Use Gemini to match and format fields with flattened JSON structure"""
        # Flatten nested JSON
        flat_json = flatten_json(json_data)

        prompt = """
        Match these flattened JSON fields to the appropriate PDF form fields and format their values:

        JSON Data:
        {}

        Available PDF Fields:
        {}

        For each JSON field:
        1. Find the best matching PDF field (ignore dots in nested field names)
        2. Format the value appropriately (empty string for None/null)
        3. For checkboxes, set True if value matches checkbox name
        4. Assign confidence based on match quality
        5. Format special fields (phones, emails, dates) appropriately

        Return a JSON object with this structure:
        {{
            "matches": [
                {{
                    "json_field": "field_name",
                    "pdf_field": "matched_pdf_field",
                    "confidence": 0.0-1.0,
                    "suggested_value": "formatted_value",
                    "reasoning": "explanation"
                }}
            ],
            "fill_data": {{
                "pdf_field_name": "formatted_value"
            }}
        }}

        Only include matches with confidence > 0.4
        Response must be valid JSON only.
        """.format(
            json.dumps(flat_json, indent=2),
            json.dumps(pdf_fields, indent=2)
        )

        try:
            response = await self.agent.run(prompt)
            result = json.loads(response.data.strip().replace("```json", "").replace("```", "").strip())


            cleaned_matches = []
            for match in result.get("matches", []):
                if all(k in match for k in ["json_field", "pdf_field", "confidence", "suggested_value", "reasoning"]):
                    match["suggested_value"] = self.preprocess_value(match["suggested_value"])
                    if match["pdf_field"] is not None:  # Additional validation
                        cleaned_matches.append(match)

            result["matches"] = cleaned_matches
            return result

        except Exception as e:
            print(f"Error in match_and_format_fields: {str(e)}")
            return {"matches": [], "fill_data": {}}

    async def process_form(
            self,
            template_pdf: str,
            json_data: Dict[str, Any],
            output_pdf: str
    ) -> ProcessingResult:
        start_time = datetime.now()

        try:
            print(f"\n📄 Processing PDF: {template_pdf}")
            if not os.path.exists(template_pdf):
                raise FileNotFoundError(f"PDF file not found: {template_pdf}")


            reader = PdfReader(template_pdf)
            pdf_fields = reader.get_form_text_fields()
            if pdf_fields is None:
                raise ValueError("No form fields found in PDF")


            pdf_field_list = []
            for page in reader.pages:
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        if annot.get_object()['/Subtype'] == '/Widget':
                            field_name = annot.get_object().get('/T')
                            if field_name:
                                pdf_field_list.append(field_name)


            print("\n🤖 Getting AI matches and formatting...")
            result = await self.match_and_format_fields(json_data, pdf_field_list)


            valid_matches = []
            for match in result["matches"]:
                try:
                    field_match = FieldMatch(**match)
                    valid_matches.append(field_match)
                except Exception as e:
                    print(f"Skipping invalid match: {str(e)}")
                    continue


            flat_json = flatten_json(json_data)
            matched_json_fields = {match.json_field for match in valid_matches}
            unmatched = [field for field in flat_json.keys() if field not in matched_json_fields]


            fill_data = {match.pdf_field: match.suggested_value for match in valid_matches}
            print(f"\n📝 Filling PDF with {len(fill_data)} matched fields")
            fillpdfs.write_fillable_pdf(template_pdf, output_pdf, fill_data)


            success_rate = (len(valid_matches) / len(flat_json)) * 100 if flat_json else 0

            result = ProcessingResult(
                status="success",
                matches=valid_matches,
                unmatched_fields=unmatched,
                success_rate=success_rate,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

            self.processing_history.append(result)
            return result

        except Exception as e:
            print(f"\n❌ Error during processing: {str(e)}")
            print(traceback.format_exc())
            return ProcessingResult(
                status="error",
                matches=[],
                unmatched_fields=list(flatten_json(json_data).keys()),
                success_rate=0,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def run(self, template_pdf: str, json_path: str, output_pdf: str):
        """Main execution method"""
        try:
            print("\n🤖 Starting Intelligent Form Processing...")

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
                    print(f"  Confidence: {match.confidence:.2f}")
                    print(f"  Value: {match.suggested_value}")
                    print(f"  Reasoning: {match.reasoning}")

            if result.unmatched_fields:
                print("\n⚠️ Unmatched Fields:")
                for field in result.unmatched_fields:
                    print(f"• {field}")

            print(f"\n💾 Filled PDF saved to: {output_pdf}")
            return result

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print(traceback.format_exc())



async def main():

    sample_data = {
        "Contact Person": {
            "First Name": "Sam",
            "Last Name": "Altman",
            "Phone Number": "(555)-(1122221)",
            "Email": "sam.altman@gpt.com"
        },
        "Entity Information": {
            "Entity Name": "red llc",
            "Entity Number": 86670,
            "Comments": "null"
        },
        "Limited Liability Company Name": "LLC",
        "Business Addresses": {
            "Initial Street Address of Principal Office": {
                "City": "Albany",
                "State": "def",
                "Zip Code": "cde"
            },
            "Initial Mailing Address of LLC, if different than item 2a": {
                "City": "ghi",
                "State": "lmn",
                "Zip Code": "599439"
            }
        },
        "Service of Process": {
            "Individual": {
                "California Agent's First Name": "redberyl",
                "Middle Name": "S",
                "Last Name": "Altman",
                "Suffix": None,
                "Street Address": {
                    "City": None,
                    "State": None,
                    "Zip Code": None
                }
            },
            "Corporation": {
                "California Registered Corporate Agent's Name": "redberyl llc"
            }
        },
        "Management": None,
        "Purpose Statement": None,
        "Organizer sign here": None,
        "Print your name here": None
    }


    json_path = "form_data.json"
    with open(json_path, "w") as f:
        json.dump(sample_data, f, indent=4)

    filler = GeminiFormFiller(API_KEY)


    result = await filler.run(
        template_pdf="California_LLC.pdf",
        json_path=json_path,
        output_pdf="filled_form.pdf"
    )

    # Print processing history
    print("\n📋 Processing History:")
    for i, hist in enumerate(filler.processing_history, 1):
        print(f"\nRun {i}:")
        print(f"Success Rate: {hist.success_rate:.1f}%")
        print(f"Execution Time: {hist.execution_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())