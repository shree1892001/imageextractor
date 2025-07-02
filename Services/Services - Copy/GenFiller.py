import asyncio
import json
import os
import re
import shutil
from typing import Dict, Any, List, Optional, Tuple

import fitz  # PyMuPDF
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *

# Replace with your actual API key
API_KEY = API_KEY_5


class FieldMatch(BaseModel):
    json_field: str
    pdf_field: str
    confidence: float
    suggested_value: Any
    reasoning: str
    semantic_similarity: float = 0.0  # Add semantic similarity score

    @field_validator("confidence")
    def validate_confidence(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return float(v)

    @field_validator("semantic_similarity")
    def validate_semantic_similarity(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Semantic similarity must be between 0 and 1")
        return float(v)


class AIPDFFieldMatcher:
    def __init__(self, api_key, debug=False):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt="""You are an expert at semantic field matching between JSON data and PDF forms.
            Your specialty is understanding the meaning and purpose of fields even when their names differ significantly.
            You use contextual understanding to accurately map fields based on their semantic meaning rather than just string similarity.
            You're particularly good at recognizing common field patterns across different naming conventions.
            Be extremely careful with fields that have UUID-style names - only match them if you have strong contextual evidence."""
        )
        self.filled_fields = []
        self.flat_json = {}  # Store flattened JSON for later use
        self.debug = debug
        self.manual_mappings = {}  # For users to provide manual field mappings

    def set_manual_mappings(self, mappings: Dict[str, str]):
        """Set manual mappings from PDF field names to JSON field names."""
        self.manual_mappings = mappings
        print(f"✅ Set {len(mappings)} manual field mappings")

    def _get_field_type_name(self, type_num):
        """Convert numeric field type to name for better AI understanding."""
        types = {
            1: "button",
            2: "radiobutton",
            3: "text",
            4: "checkbox",
            5: "dropdown",
            6: "listbox"
        }
        return types.get(type_num, f"unknown({type_num})")

    def _is_uuid_field(self, field_name):
        """Check if a field name is a UUID style identifier."""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, field_name, re.IGNORECASE))

    async def extract_pdf_fields(self, pdf_path: str) -> Dict[str, Dict[str, Any]]:
        """Extracts all fillable fields from a PDF."""
        print("🔍 Extracting all fillable fields...")
        doc = fitz.open(pdf_path)
        fields = {}

        # Extract field context by looking at nearby text
        field_contexts = self._extract_field_contexts(doc)

        for page_num, page in enumerate(doc):
            for widget in page.widgets():
                if widget.field_name:
                    field_name = widget.field_name.strip()
                    field_type = widget.field_type
                    field_rect = widget.rect
                    field_flags = widget.field_flags

                    # Note if this is a UUID-style field name
                    is_uuid = self._is_uuid_field(field_name)

                    fields[field_name] = {
                        "page_num": page_num,
                        "type": field_type,
                        "type_name": self._get_field_type_name(field_type),
                        "rect": [field_rect.x0, field_rect.y0, field_rect.x1, field_rect.y1],
                        "flags": field_flags,
                        "is_readonly": bool(field_flags & 1),
                        "current_value": widget.field_value,
                        "context": field_contexts.get(field_name, ""),  # Add context info
                        "is_uuid": is_uuid  # Flag UUID-style fields
                    }

        print(f"✅ Extracted {len(fields)} fields across {len(doc)} pages.")
        for field, info in fields.items():
            readonly_status = "READ-ONLY" if info["is_readonly"] else "EDITABLE"
            uuid_status = "UUID-STYLE" if info["is_uuid"] else ""
            print(
                f" - Field: '{field}' (Page {info['page_num'] + 1}) [{readonly_status}] [{info['type_name']}] {uuid_status}")
            if info["context"]:
                context_preview = info["context"][:50] + ("..." if len(info["context"]) > 50 else "")
                print(f"   Context: '{context_preview}'")

        doc.close()
        return fields

    def _extract_field_contexts(self, doc) -> Dict[str, str]:
        """Extract contextual information around form fields with improved precision."""
        contexts = {}

        for page_num, page in enumerate(doc):
            # Get text with more details for better context
            text_dict = page.get_text("dict")

            # Get all form fields on the page
            widgets = page.widgets()

            for widget in widgets:
                if not widget.field_name:
                    continue

                field_name = widget.field_name.strip()
                field_rect = widget.rect

                # Find text blocks near this field with improved context extraction
                nearby_text = []

                # Process each block of text
                for block in text_dict["blocks"]:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        line_rect = fitz.Rect(line["bbox"])
                        line_dist = self._rect_distance(field_rect, line_rect)

                        # If line is very close to field, add its text
                        if line_dist < 50:  # Closer proximity for more relevant context
                            for span in line["spans"]:
                                span_text = span["text"].strip()
                                if span_text:
                                    nearby_text.append(span_text)

                # Also look for labels that might be above or to the left of the field
                for block in text_dict["blocks"]:
                    if "lines" not in block:
                        continue

                    block_rect = fitz.Rect(block["bbox"])

                    # Check if this block is likely a label (above or to the left)
                    is_left_of = (block_rect.x1 < field_rect.x0 and
                                  block_rect.y0 < field_rect.y1 and
                                  block_rect.y1 > field_rect.y0)

                    is_above = (block_rect.y1 < field_rect.y0 and
                                block_rect.x1 > field_rect.x0 and
                                block_rect.x0 < field_rect.x1)

                    if is_left_of or is_above:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                span_text = span["text"].strip()
                                if span_text and span_text not in nearby_text:
                                    nearby_text.append(span_text)

                # Join all nearby text, but prioritize the labels
                contexts[field_name] = " ".join(nearby_text)

        return contexts

    def _rect_distance(self, rect1: fitz.Rect, rect2: fitz.Rect) -> float:
        """Calculate the distance between two rectangles."""
        # If rectangles overlap, distance is 0
        if rect1.intersects(rect2):
            return 0

        # Calculate distances between edges
        dx = max(0, max(rect1.x0 - rect2.x1, rect2.x0 - rect1.x1))
        dy = max(0, max(rect1.y0 - rect2.y1, rect2.y0 - rect1.y1))

        return (dx ** 2 + dy ** 2) ** 0.5

    def flatten_json(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested JSON objects for easier matching."""
        items = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                items.update(self.flatten_json(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(self.flatten_json(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                # Clean and normalize values
                if isinstance(value, str):
                    value = value.strip()
                items[new_key] = value

        return items

    def _extract_field_semantic_info(self, pdf_fields: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract semantic information about PDF fields."""
        field_info = {}

        for field_name, info in pdf_fields.items():
            # Break down field name into components
            components = re.split(r'[._\- ]', field_name)

            # Extract potential semantic meaning
            field_type_indicators = {
                "name": ["name", "firstName", "lastName", "fullName"],
                "address": ["address", "street", "city", "state", "zip", "postal"],
                "contact": ["phone", "email", "fax", "contact"],
                "date": ["date", "dateOf", "dob"],
                "identification": ["id", "ssn", "ein", "tax"]
            }

            detected_categories = []
            # Only try to detect categories if it's not a UUID
            if not info["is_uuid"]:
                for component in components:
                    component = component.lower()
                    for category, indicators in field_type_indicators.items():
                        if any(indicator in component for indicator in indicators):
                            detected_categories.append(category)

            # For UUID fields, try to detect categories from context
            elif info["context"]:
                context_lower = info["context"].lower()
                for category, indicators in field_type_indicators.items():
                    if any(indicator in context_lower for indicator in indicators):
                        detected_categories.append(category)

            field_info[field_name] = {
                "components": components,
                "categories": detected_categories,
                "context": info.get("context", ""),
                "field_type": info["type"],
                "field_type_name": info["type_name"],
                "is_uuid": info["is_uuid"]
            }

        return field_info

    async def match_fields_with_ai(self, pdf_fields: Dict[str, Dict[str, Any]],
                                   json_data: Dict[str, Any]) -> List[FieldMatch]:
        """Use AI to match JSON fields with PDF form fields using semantic understanding."""
        print("🧠 Using AI for semantic field matching...")

        # Flatten JSON data and store for later use
        self.flat_json = self.flatten_json(json_data)

        # Handle any manual mappings first
        matches = []
        for pdf_field, json_field in self.manual_mappings.items():
            if pdf_field in pdf_fields and json_field in self.flat_json:
                matches.append(FieldMatch(
                    json_field=json_field,
                    pdf_field=pdf_field,
                    confidence=1.0,  # Maximum confidence for manual mappings
                    suggested_value=self.flat_json[json_field],
                    reasoning="Manual mapping provided by user",
                    semantic_similarity=1.0
                ))
                print(f"✅ Using manual mapping: {json_field} → {pdf_field}")

        # Extract additional semantic information about fields
        pdf_field_semantics = self._extract_field_semantic_info(pdf_fields)

        # Prepare sample values for better understanding
        sample_json_values = {}
        for k in list(self.flat_json.keys())[:30]:  # Limit to first 30 fields for conciseness
            value = self.flat_json[k]
            # Truncate long string values
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            sample_json_values[k] = value

        # Create a prompt for the AI with enhanced guidance for better matching
        prompt = f"""
        Your task is to semantically match JSON data fields to PDF form fields, even when their names differ significantly.

        # JSON Data Fields (Flattened) with Sample Values
        {json.dumps(sample_json_values, indent=2)}

        # PDF Form Fields with Context
        {json.dumps([{
            "field_name": k,
            "type": v["type_name"],
            "context": v.get("context", "")[:200],
            "is_uuid": v["is_uuid"],
            "components": pdf_field_semantics[k]["components"],
            "categories": pdf_field_semantics[k]["categories"]
        } for k, v in pdf_fields.items()], indent=2)}

        IMPORTANT RULES:
        1. Focus on matching as many fields as possible, even if there's only partial confidence.
        2. For PDF fields with UUID-style names (where is_uuid=true), rely ONLY on the context text, not the field name.
        3. Never match the same JSON field to multiple PDF fields.
        4. For checkbox fields, only suggest boolean or boolean-convertible values.
        5. For text fields, ensure the value type is appropriate.
        6. It's better to provide more potential matches at lower confidence than to miss important ones.
        7. For fields with common business terms (company name, address, state, etc.), be more aggressive in matching.

        For each PDF field, find the most semantically appropriate JSON field that would provide its value.
        Pay special attention to:

        1. Semantic meaning of fields (e.g., "customerName" and "applicantName" might refer to the same concept)
        2. Field context from surrounding text
        3. Common patterns in form fields (address fields, contact info, dates, identification numbers)
        4. Field name components and their possible meanings
        5. Data types and expected formats

        Please return your matches in this JSON format:
        ```json
        {{
          "matches": [
            {{
              "json_field": "string",
              "pdf_field": "string", 
              "confidence": 0.0-1.0,
              "suggested_value": "string",
              "reasoning": "string",
              "semantic_similarity": 0.0-1.0
            }}
          ]
        }}
        ```

        For each match:
        - Explain your reasoning in detail 
        - Include a semantic similarity score (0-1) that reflects how conceptually similar the fields are
        - For fields with low string similarity but high semantic similarity, explain the semantic connection
        - Consider field context and purpose, not just the field name
        - For UUID-style fields, give a lower confidence score, but still suggest matches where context provides clues

        Only include JSON in your response, no additional text.
        """

        # Call the AI agent
        response = await self.agent.run(prompt)

        # Parse the AI response
        ai_matches = self.parse_ai_response(response.data, self.flat_json)

        # Add AI-generated matches that don't conflict with manual mappings
        already_matched_json_fields = {m.json_field for m in matches}
        already_matched_pdf_fields = {m.pdf_field for m in matches}

        for match in ai_matches:
            if (match.json_field not in already_matched_json_fields and
                    match.pdf_field not in already_matched_pdf_fields):
                matches.append(match)
                already_matched_json_fields.add(match.json_field)
                already_matched_pdf_fields.add(match.pdf_field)

        # Return matches sorted by confidence
        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    def parse_ai_response(self, response_text: str, flat_json: Dict[str, Any]) -> List[FieldMatch]:
        """Parse the AI response and extract field matches."""
        print("Parsing AI semantic matching response...")

        # Try to extract JSON from the response
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response_text)

        if json_match:
            response_text = json_match.group(1).strip()

        try:
            # Parse the JSON response
            data = json.loads(response_text)
            matches = []

            # Process field matches
            match_list = data.get("matches", [])
            for match in match_list:
                # Get the actual value from our flattened JSON
                json_field = match["json_field"]

                # FIX: Check if the json_field exists in flat_json before accessing it
                if json_field in flat_json:
                    match["suggested_value"] = flat_json[json_field]
                else:
                    # If the key doesn't exist, use a default or empty value
                    print(f"⚠️ Warning: JSON field '{json_field}' not found in data - using empty value")
                    match["suggested_value"] = ""

                # Set default values if missing
                match.setdefault("confidence", 0.8)
                match.setdefault("reasoning", "Matched by semantic similarity")
                match.setdefault("semantic_similarity", 0.5)

                try:
                    validated_match = FieldMatch(**match)
                    matches.append(validated_match)
                except Exception as e:
                    print(f"⚠️ Invalid field match: {e}")

            return matches

        except json.JSONDecodeError as e:
            print(f"❌ Error parsing AI response: {e}")
            print(f"Response text: {response_text[:200]}...")
            return []

    def convert_to_boolean(self, value):
        """Convert various value formats to boolean for checkboxes."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.lower()
            return value in ['true', 'yes', 'y', '1', 'checked', 'selected']
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    async def analyze_unmatched_fields(self, unmatched_pdf_fields: List[str], unmatched_json_fields: List[str],
                                       pdf_fields: Dict[str, Dict[str, Any]]) -> List[FieldMatch]:
        """Use AI to analyze unmatched fields and suggest alternative matches."""
        if not unmatched_pdf_fields or not unmatched_json_fields:
            return []

        print(f"🔍 Analyzing {len(unmatched_pdf_fields)} unmatched PDF fields...")

        # Prepare field info for unmatched PDF fields
        unmatched_pdf_info = []
        for name in unmatched_pdf_fields[:30]:  # Increased limit to 30 fields
            if name in pdf_fields:
                field = pdf_fields[name]
                unmatched_pdf_info.append({
                    "name": name,
                    "type": field["type_name"],
                    "context": field.get("context", "")[:200],
                    "is_uuid": field["is_uuid"]
                })

        # Prepare sample values for unmatched JSON fields
        unmatched_json_values = {}
        for k in unmatched_json_fields[:30]:  # Increased limit to 30 fields
            if k in self.flat_json:
                value = self.flat_json[k]
                # Truncate long string values
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                unmatched_json_values[k] = value

        # Enhanced prompt for more aggressive matching
        prompt = f"""
        You need to find matches between these unmatched fields by understanding their semantic meaning.
        We need as many good matches as possible, even if not perfect.

        # Unmatched PDF Fields
        {json.dumps(unmatched_pdf_info, indent=2)}

        # Unmatched JSON Fields with Values
        {json.dumps(unmatched_json_values, indent=2)}

        IMPORTANT RULES:
        1. Focus on finding as many potential matches as possible - be creative and thorough.
        2. For PDF fields with UUID-style names (where is_uuid=true), rely ONLY on the context text, not the field name.
        3. Never match the same JSON field to multiple PDF fields.
        4. For checkbox fields, only suggest boolean or boolean-convertible values.
        5. For text fields, ensure the value type is appropriate.
        6. We want comprehensive field matching, even at lower confidence levels.
        7. Common business form concepts should be matched aggressively (company names, addresses, states, dates, etc.)

        For each PDF field, suggest the most semantically similar JSON field, even if the names are completely different.
        Focus on what the fields are meant to represent, not their literal names.

        Please return your matches in this JSON format:
        ```json
        {{
          "matches": [
            {{
              "json_field": "string",
              "pdf_field": "string", 
              "confidence": 0.0-1.0,
              "suggested_value": "string",
              "reasoning": "detailed explanation of the semantic relationship",
              "semantic_similarity": 0.0-1.0
            }}
          ]
        }}
        ```

        Include matches even at lower confidence levels (>0.4) if there's any reasonable semantic relationship.
        Be particularly thorough with business entity information like names, addresses, states, etc.

        Return only JSON, no additional text.
        """

        response = await self.agent.run(prompt)
        matches = self.parse_ai_response(response.data, self.flat_json)

        # Less restrictive filtering for UUID fields to get more matches
        return [m for m in matches if (m.confidence > 0.4)]

    async def fill_pdf_with_matches(self, pdf_path: str, matches: List[FieldMatch],
                                    pdf_fields: Dict[str, Dict[str, Any]], output_pdf: str) -> bool:
        """Fill PDF form fields with matched values."""
        print("✍️ Filling PDF with matched values...")

        try:
            doc = fitz.open(pdf_path)
            filled_count = 0
            unmatched_pdf_fields = set(pdf_fields.keys())
            used_json_fields = set()  # Track which JSON fields we've already used

            # First pass - fill fields with high confidence matches
            high_confidence_matches = [m for m in matches if m.confidence >= 0.7]  # Lower threshold to get more matches
            for match in high_confidence_matches:
                field_name = match.pdf_field
                json_field = match.json_field

                # Skip if we've already used this JSON field (prevent duplication)
                if json_field in used_json_fields:
                    if self.debug:
                        print(f"⚠️ Skipping duplicate use of JSON field: {json_field}")
                    continue

                if field_name in unmatched_pdf_fields:
                    unmatched_pdf_fields.remove(field_name)

                if field_name not in pdf_fields:
                    continue

                if self._fill_field(doc, field_name, match.suggested_value, pdf_fields[field_name]):
                    filled_count += 1
                    self.filled_fields.append(field_name)
                    used_json_fields.add(json_field)  # Mark this JSON field as used

                    # Flag if this is a UUID field
                    uuid_flag = " [UUID]" if pdf_fields[field_name]["is_uuid"] else ""
                    print(
                        f"✅ Filled (high confidence): '{match.suggested_value}' → '{field_name}'{uuid_flag} (Confidence: {match.confidence:.2f})")

                    # Debug: show reasoning
                    if self.debug:
                        print(f"   Reasoning: {match.reasoning[:100]}...")

            # Get list of remaining JSON fields
            unmatched_json_fields = [f for f in self.flat_json.keys() if f not in used_json_fields]

            # Try to find matches for remaining fields
            if unmatched_pdf_fields and unmatched_json_fields:
                additional_matches = await self.analyze_unmatched_fields(
                    list(unmatched_pdf_fields), unmatched_json_fields, pdf_fields
                )

                # Fill fields with additional matches
                for match in additional_matches:
                    field_name = match.pdf_field
                    json_field = match.json_field

                    # Skip if we've already used this JSON field (prevent duplication)
                    if json_field in used_json_fields:
                        if self.debug:
                            print(f"⚠️ Skipping duplicate use of JSON field: {json_field}")
                        continue

                    if field_name in unmatched_pdf_fields:
                        unmatched_pdf_fields.remove(field_name)

                    if field_name not in pdf_fields:
                        continue

                    if self._fill_field(doc, field_name, match.suggested_value, pdf_fields[field_name]):
                        filled_count += 1
                        self.filled_fields.append(field_name)
                        used_json_fields.add(json_field)  # Mark this JSON field as used

                        # Flag if this is a UUID field
                        uuid_flag = " [UUID]" if pdf_fields[field_name]["is_uuid"] else ""
                        print(
                            f"✅ Filled (semantic match): '{match.suggested_value}' → '{field_name}'{uuid_flag} (Similarity: {match.semantic_similarity:.2f})")

                        # Debug: show reasoning
                        if self.debug:
                            print(f"   Reasoning: {match.reasoning[:100]}...")

            # Save the modified PDF
            doc.save(output_pdf, deflate=True, clean=True, garbage=4, pretty=False)
            doc.close()

            print(f"✅ Successfully filled {filled_count} fields out of {len(pdf_fields)} available")
            return filled_count > 0

        except Exception as e:
            print(f"❌ Error filling PDF: {e}")
            return False

    def _fill_field(self, doc, field_name: str, value: Any, field_info: Dict) -> bool:
        """Helper method to fill a single field with improved type checking."""
        try:
            if field_info["is_readonly"]:
                print(f"⚠️ Skipping readonly field '{field_name}'")
                return False

            field_type = field_info["type"]

            # Validate value is appropriate for field type
            if field_type == 4:  # Checkbox
                value = self.convert_to_boolean(value)
            elif field_type == 3:  # Text
                if not isinstance(value, (str, int, float)):
                    print(f"⚠️ Invalid value type for text field '{field_name}': {type(value)}")
                    return False
                value = str(value)
            elif field_type == 5 or field_type == 6:  # Dropdown or listbox
                # Ensure value is in the list of choices
                widget = None
                for w in doc[field_info["page_num"]].widgets():
                    if w.field_name == field_name:
                        widget = w
                        break

                if widget and hasattr(widget, 'choice_values'):
                    choices = widget.choice_values
                    if choices and value not in choices:
                        print(f"⚠️ Value '{value}' not in choices for field '{field_name}'")
                        return False

            # Apply the value to the field
            page_num = field_info["page_num"]
            page = doc[page_num]

            for widget in page.widgets():
                if widget.field_name == field_name:
                    widget.field_value = value
                    widget.update()
                    return True

            return False
        except Exception as e:
            print(f"⚠️ Error filling {field_name}: {e}")
            return False

    def verify_pdf_filled(self, pdf_path: str) -> bool:
        """Verify that fields were filled in the PDF."""
        try:
            doc = fitz.open(pdf_path)
            filled_fields = []

            for page in doc:
                for widget in page.widgets():
                    if widget.field_value:
                        filled_fields.append(widget.field_name)

            doc.close()

            if filled_fields:
                print(f"✅ Verified {len(filled_fields)} filled form fields")
                sample = filled_fields[:5]
                print("Sample filled fields:", sample)
                return True
            else:
                print("❌ No filled fields found in the PDF")
                return False

        except Exception as e:
            print(f"❌ Error verifying PDF: {e}")
            return False

    async def match_and_fill_pdf(self, pdf_path: str, json_data: Dict[str, Any], output_pdf: str) -> bool:
        """Match JSON fields to PDF fields using semantic AI matching and fill the PDF."""
        # Create backup of original PDF
        backup_pdf = f"{pdf_path}.backup"
        shutil.copy2(pdf_path, backup_pdf)
        print(f"Created backup of original PDF: {backup_pdf}")

        # Extract PDF fields
        pdf_fields = await self.extract_pdf_fields(pdf_path)

        # We're using AI matching instead of manual mappings
        # Set enhanced system prompt to encourage more matches
        self.agent.system_prompt = """You are an expert at semantic field matching between JSON data and PDF forms.
        Your specialty is understanding the meaning and purpose of fields even when their names differ significantly.
        You use contextual understanding to accurately map fields based on their semantic meaning rather than just string similarity.
        You're particularly good at recognizing common field patterns across different naming conventions.
        Be thorough and try to match as many fields as possible.
        For UUID-style fields, use any contextual clues available and don't be afraid to suggest matches when they make logical sense.
        It's better to suggest more potential matches than to miss important ones."""

        # Match fields with AI
        matches = await self.match_fields_with_ai(pdf_fields, json_data)

        if not matches:
            print("❌ No field matches found by AI")
            return False

        print(f"🔍 AI found {len(matches)} field matches")
        # Print some sample matches
        for match in matches[:5]:
            uuid_flag = " [UUID]" if pdf_fields[match.pdf_field]["is_uuid"] else ""
            print(
                f" - {match.json_field} → {match.pdf_field}{uuid_flag} (Confidence: {match.confidence:.2f}, Semantic: {match.semantic_similarity:.2f})")
            print(f"   Value: {match.suggested_value}")
            if self.debug:
                print(f"   Reasoning: {match.reasoning}")

        # Fill PDF with matches
        success = await self.fill_pdf_with_matches(pdf_path, matches, pdf_fields, output_pdf)

        # Verify PDF was filled
        if success:
            return self.verify_pdf_filled(output_pdf)
        else:
            return False


async def main():
    # Paths to files
    template_pdf = "D:\\demo\\Services\\WisconsinCorp.pdf"
    json_path = "D:\\demo\\Services\\form_data.json"
    output_pdf = "D:\\demo\\Services\\fill_smart14.pdf"

    # Set debug mode
    debug_mode = True

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Process PDF with AI matching
    print(f"🚀 Processing PDF: {template_pdf}")
    print(f"📄 Using JSON data from: {json_path}")
    print(f"💾 Output will be saved to: {output_pdf}")

    matcher = AIPDFFieldMatcher(API_KEY, debug=debug_mode)

    # Let's modify the system prompt to encourage more matches
    matcher.agent.system_prompt = """You are an expert at semantic field matching between JSON data and PDF forms.
    Your specialty is understanding the meaning and purpose of fields even when their names differ significantly.
    You use contextual understanding to accurately map fields based on their semantic meaning rather than just string similarity.
    You're particularly good at recognizing common field patterns across different naming conventions.
    Be thorough and try to match as many fields as possible.
    For UUID-style fields, use any contextual clues available and don't be afraid to suggest matches when they make logical sense.
    It's better to suggest more potential matches than to miss important ones."""

    # Print available JSON fields to help with debugging
    print("\n📋 Available JSON fields:")
    flattened_json = matcher.flatten_json(json_data)
    for key, value in flattened_json.items():
        print(f" - {key}: {value}")

    # Execute the matching and filling process
    try:
        # Extract fields first to make them available for inspection
        pdf_fields = await matcher.extract_pdf_fields(template_pdf)

        # Add some manual mappings for critical fields based on what we see in the JSON
        # This helps with fields that might be harder for the AI to match
        # Replace these with actual field mappings from your form
        matcher.set_manual_mappings({
            # Example mappings - replace with actual field names from your PDF
            # "pdf_field_name": "json_field_name",
            # "corporation_name": "data.EntityName",
            # "state_of_incorporation": "data.State.stateFullDesc"
        })

        # Continue with matching and filling
        matches = await matcher.match_fields_with_ai(pdf_fields, json_data)

        print(f"\n🔍 AI found {len(matches)} field matches")
        # Print all matches to help diagnose issues
        for match in matches:
            uuid_flag = " [UUID]" if pdf_fields[match.pdf_field]["is_uuid"] else ""
            print(
                f" - {match.json_field} → {match.pdf_field}{uuid_flag} (Confidence: {match.confidence:.2f}, Semantic: {match.semantic_similarity:.2f})")
            print(f"   Value: {match.suggested_value}")
            if debug_mode:
                print(f"   Reasoning: {match.reasoning}")

        # Lower the confidence threshold for second-pass field analysis
        if len(matches) < len(pdf_fields) / 3:  # If we matched less than 1/3 of fields
            print("\n⚠️ Found very few matches. Trying with lower confidence threshold...")

            # Get unmatched fields
            matched_pdf_fields = {m.pdf_field for m in matches}
            unmatched_pdf_fields = [field for field in pdf_fields.keys() if field not in matched_pdf_fields]

            # Get all JSON fields
            all_json_fields = list(flattened_json.keys())
            matched_json_fields = {m.json_field for m in matches}
            unmatched_json_fields = [field for field in all_json_fields if field not in matched_json_fields]

            # Try more aggressive matching
            additional_matches = await matcher.analyze_unmatched_fields(
                unmatched_pdf_fields, unmatched_json_fields, pdf_fields
            )

            if additional_matches:
                print(f"🔍 Found {len(additional_matches)} additional matches with lower confidence")
                matches.extend(additional_matches)

        # Fill the PDF with the matches
        try:
            success = await matcher.fill_pdf_with_matches(template_pdf, matches, pdf_fields, output_pdf)
        except Exception as e:
            print(f"❌ Error during PDF filling: {e}")

            # Try to identify the problematic field
            print("\n⚠️ Attempting to identify problematic field...")

            # Create a new output file path
            troubleshoot_pdf = output_pdf.replace(".pdf", "_troubleshoot.pdf")

            # Try filling fields one by one
            doc = fitz.open(template_pdf)
            filled_count = 0
            error_fields = []

            for match in matches:
                try:
                    field_name = match.pdf_field
                    if field_name in pdf_fields:
                        field_info = pdf_fields[field_name]
                        if matcher._fill_field(doc, field_name, match.suggested_value, field_info):
                            filled_count += 1
                            print(f"✅ Successfully filled: {field_name}")
                        else:
                            print(f"⚠️ Could not fill: {field_name}")
                except Exception as field_error:
                    print(f"❌ Error with field {match.pdf_field}: {field_error}")
                    error_fields.append(match.pdf_field)

            if filled_count > 0:
                # Save what we managed to fill
                doc.save(troubleshoot_pdf, deflate=True, clean=True, garbage=4, pretty=False)
                print(f"✅ Partially filled PDF saved to: {troubleshoot_pdf}")
                print(f"✅ Successfully filled {filled_count} out of {len(matches)} matched fields")

            if error_fields:
                print(f"❌ Problem fields: {error_fields}")

            doc.close()
            return filled_count > 0

        if success:
            print("✅ PDF form filling completed successfully!")
        else:
            print("❌ PDF form filling failed.")

        return success
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())