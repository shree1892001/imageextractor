import json
from typing import Dict, List, Optional, Tuple

class PDFFormFiller:
    def __init__(self, json_data: Dict, pdf_fields: Dict, ocr_elements: List[Dict], field_context: Dict):
        self.json_data = json_data
        self.pdf_fields = pdf_fields
        self.ocr_elements = ocr_elements
        self.field_context = field_context
        self.matches = []
        self.ocr_matches = []
        self.current_page = 1
        self.page_fields = self._group_fields_by_page()

    def _group_fields_by_page(self) -> Dict[int, Dict]:
        """Group PDF form fields by their page number."""
        page_fields = {}
        for field_id, field_info in self.pdf_fields.items():
            page_num = field_info.get('page', 1)
            if page_num not in page_fields:
                page_fields[page_num] = {}
            page_fields[page_num][field_id] = field_info
        return page_fields

    def _fill_form_fields(self) -> None:
        """Match and fill form fields page by page."""
        for page_num in sorted(self.page_fields.keys()):
            self.current_page = page_num
            page_pdf_fields = self.page_fields[page_num]
            
            # Entity Name matching for current page
            entity_name = self._get_entity_name()
            if entity_name:
                for field_id, field_info in page_pdf_fields.items():
                    if any(name in field_info['name'].lower() for name in ['entity name', 'llc name', 'limited liability company', 'corporation']):
                        self._match_field(field_id, 'entity_name', entity_name)

            # Registered Agent matching for current page
            agent_name = self._get_registered_agent_name()
            if agent_name:
                for field_id, field_info in page_pdf_fields.items():
                    if any(term in field_info['name'].lower() for term in ['registered agent', 'commercial registered agent', 'initial registered agent']):
                        self._match_field(field_id, 'registered_agent_name', agent_name)

            # Principal Address matching for current page
            address = self._get_principal_address()
            if address:
                for field_id, field_info in page_pdf_fields.items():
                    field_name = field_info['name'].lower()
                    if 'principal' in field_name and 'address' in field_name:
                        if 'line1' in field_name or 'street' in field_name:
                            self._match_field(field_id, 'principal_address_line1', address['address_line1'])
                        elif 'city' in field_name:
                            self._match_field(field_id, 'principal_address_city', address['city'])
                        elif 'state' in field_name:
                            self._match_field(field_id, 'principal_address_state', address['state'])
                        elif 'zip' in field_name:
                            self._match_field(field_id, 'principal_address_zip', address['zip_code'])

            # Organizer Details matching for current page
            organizer = self._get_organizer_details()
            if organizer:
                for field_id, field_info in page_pdf_fields.items():
                    field_name = field_info['name'].lower()
                    if 'organizer' in field_name:
                        if 'name' in field_name or 'signature' in field_name:
                            self._match_field(field_id, 'organizer_name', organizer['name'])
                        elif 'email' in field_name:
                            self._match_field(field_id, 'organizer_email', organizer['email'])

            # Contact Details matching for current page
            contact = self._get_contact_details()
            if contact:
                for field_id, field_info in page_pdf_fields.items():
                    field_name = field_info['name'].lower()
                    if 'name' in field_name and not any(x in field_name for x in ['entity', 'company', 'organizer']):
                        full_name = f"{contact['first_name']} {contact['last_name']}".strip()
                        self._match_field(field_id, 'contact_name', full_name)
                    elif 'email' in field_name and not 'organizer' in field_name:
                        self._match_field(field_id, 'contact_email', contact['email'])

    def _get_entity_name(self) -> Optional[str]:
        """Get entity name from JSON data checking multiple possible paths."""
        possible_paths = [
            'entity_name',
            'llc_name',
            'Corporation_Name',
            'Corp_Name'
        ]
        for path in possible_paths:
            if path in self.json_data:
                return self.json_data[path]
        return None

    def _get_registered_agent_name(self) -> Optional[str]:
        """Get registered agent name from JSON data."""
        try:
            return self.json_data['data']['orderDetails']['strapiOrderFormJson']['Payload']['Entity_Formation']['Registered_Agent']['RA_Name']
        except KeyError:
            return None

    def _get_principal_address(self) -> Dict:
        """Get principal address details from JSON data."""
        try:
            address_data = self.json_data['data']['orderDetails']['strapiOrderFormJson']['Payload']['Entity_Formation']['Principal_Address']
            return {
                'address_line1': address_data.get('PA_Address_Line_1', ''),
                'address_line2': address_data.get('PA_Address_Line_2', ''),
                'city': address_data.get('PA_City', ''),
                'state': address_data.get('PA_State', ''),
                'zip_code': address_data.get('PA_Zip_Code', '')
            }
        except KeyError:
            return {}

    def _get_organizer_details(self) -> Dict:
        """Get organizer details from JSON data."""
        try:
            org_data = self.json_data['data']['orderDetails']['strapiOrderFormJson']['Payload']['Entity_Formation']['Organizer_Information']
            return {
                'name': org_data['Organizer_Details'].get('Org_Name', ''),
                'email': org_data['Organizer_Details'].get('Organizer_Email', ''),
                'address_line1': org_data['Organizer_Address'].get('Org_Address_Line1', ''),
                'state': org_data['Organizer_Address'].get('Org_State', ''),
                'zip_code': org_data['Organizer_Address'].get('Org_Zip_Code', '')
            }
        except KeyError:
            return {}

    def _get_contact_details(self) -> Dict:
        """Get contact details from JSON data."""
        try:
            contact_data = self.json_data['data']['contactDetails']
            return {
                'first_name': contact_data.get('firstName', ''),
                'last_name': contact_data.get('lastName', ''),
                'email': contact_data.get('emailId', '')
            }
        except KeyError:
            return {}

    def _match_field(self, pdf_field: str, field_type: str, value: str, confidence: float = 0.9) -> None:
        """Add a field match to the matches list."""
        self.matches.append({
            'json_field': field_type,
            'pdf_field': pdf_field,
            'confidence': confidence,
            'suggested_value': value,
            'reasoning': f'Matched based on field type: {field_type}'
        })

    def process(self) -> Dict:
        """Process the PDF form and return matches."""
        self._fill_form_fields()
        return {
            'matches': self.matches,
            'ocr_matches': self.ocr_matches
        }

    def fill_form_fields(self, json_data: Dict[str, Any], pdf_fields: Dict[str, Any]) -> List[FieldMatch]:
        """Fill form fields with matched data from JSON."""
        try:
            if not pdf_fields:
                raise ValueError("pdf_fields argument is required")
            self.matches = self._match_fields(json_data, pdf_fields)
            return self.matches
        except Exception as e:
            raise Exception(f"Error filling form fields: {str(e)}")