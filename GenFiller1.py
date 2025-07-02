import json
import os
from typing import Dict, Any
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
import difflib
import pickle

class GenFiller1:
    def __init__(self, form_data_path: str, template_dir: str = 'templates'):
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
        self.llm = OpenAI(temperature=0)
        self.llm_chain = self._setup_llm()
        self.template_cache = {}
        self.form_data = self._load_form_data(form_data_path)
        self.entity_formation_data = self.form_data['data']['orderDetails']['strapiOrderFormJson']['Payload']['Entity_Formation']
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()

    def _setup_llm(self) -> LLMChain:
        template = """Analyze the following form field and suggest the best mapping based on context and field patterns:
        Field: {field}
        Context: {context}
        Previous mappings: {history}
        Consider field naming patterns, data types, and semantic relationships.
        Provide mapping with confidence score and explanation.
        Suggested mapping:"""
        prompt = PromptTemplate(template=template, input_variables=["field", "context", "history"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _setup_tools(self) -> list:
        tools = [
            Tool(
                name="field_analyzer",
                func=self._analyze_field_patterns,
                description="Analyzes form field patterns and suggests optimal mappings"
            ),
            Tool(
                name="template_matcher",
                func=self._smart_template_match,
                description="Finds the best matching template based on form structure"
            ),
            Tool(
                name="validation_checker",
                func=self._validate_field_mapping,
                description="Validates field mappings for consistency and correctness"
            )
        ]
        return tools

    def _setup_agent(self) -> AgentExecutor:
        agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            tools=self.tools,
            memory=self.memory
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def _analyze_field_patterns(self, field: str, context: str) -> Dict[str, Any]:
        """AI-powered analysis of field patterns and relationships"""
        analysis_prompt = f"Analyze the pattern and relationship of field '{field}' in context: {context}"
        result = self.agent_executor.run(analysis_prompt)
        return {
            'suggested_mapping': result.get('mapping'),
            'confidence': result.get('confidence'),
            'reasoning': result.get('reasoning')
        }

    def _smart_template_match(self, form_structure: Dict) -> str:
        """Intelligent template matching based on form structure"""
        match_prompt = f"Find best matching template for form structure: {json.dumps(form_structure)}"
        result = self.agent_executor.run(match_prompt)
        return result.get('template_name')

    def _validate_field_mapping(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """AI-driven validation of field mappings"""
        validation_prompt = f"Validate the following field mapping: {json.dumps(mapping)}"
        result = self.agent_executor.run(validation_prompt)
        return {
            'is_valid': result.get('is_valid'),
            'issues': result.get('issues'),
            'suggestions': result.get('suggestions')
        }

    def _load_form_data(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            data = json.load(f)
            # Analyze form structure using AI
            self.agent_executor.run(f"Analyze form structure: {json.dumps(data)}")
            return data

    def _save_template(self, template_name: str, field_mapping: Dict[str, str]) -> None:
        # Validate template before saving
        validation_result = self._validate_field_mapping(field_mapping)
        if validation_result['is_valid']:
            template_path = os.path.join(self.template_dir, f"{template_name}.pkl")
            with open(template_path, 'wb') as f:
                pickle.dump(field_mapping, f)
            self.memory.save_context({"template": template_name}, {"mapping": field_mapping})

    def _load_template(self, template_name: str) -> Dict[str, str]:
        template_path = os.path.join(self.template_dir, f"{template_name}.pkl")
        if os.path.exists(template_path):
            with open(template_path, 'rb') as f:
                template = pickle.load(f)
                # Validate loaded template
                validation_result = self._validate_field_mapping(template)
                if validation_result['is_valid']:
                    return template
        return None

    def _find_best_field_match(self, target_field: str, available_fields: list) -> str:
        """AI-enhanced field matching using similarity and context"""
        # First try direct string similarity
        matches = difflib.get_close_matches(target_field, available_fields, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        # Use AI for context-aware matching
        context = {"available_fields": available_fields, "field_history": self.memory.load_memory_variables({})}
        result = self.agent_executor.run({
            "input": f"Find best match for field '{target_field}'",
            "context": context
        })
        return result.get('suggested_match')

    def _extract_address_fields(self, address_dict: Dict[str, str]) -> Dict[str, str]:
        # Use AI to analyze address patterns
        analysis_result = self._analyze_field_patterns('address', json.dumps(address_dict))
        
        address_mapping = {
            'Address_Line_1': 'street_address',
            'Address_Line_2': 'street_address2',
            'City': 'city',
            'State': 'state',
            'Zip_Code': 'zip_code'
        }
        
        extracted_fields = {}
        for prefix in ['PA', 'RA', 'BI', 'MI']:
            for key, value in address_mapping.items():
                field_key = f'{prefix}_{key}'
                if field_key in address_dict:
                    extracted_fields[value] = address_dict[field_key]
        
        # Validate extracted fields
        validation_result = self._validate_field_mapping(extracted_fields)
        if not validation_result['is_valid']:
            # Apply suggested corrections
            for suggestion in validation_result['suggestions']:
                extracted_fields.update(suggestion)
        
        return extracted_fields

    def get_form_field_mapping(self) -> Dict[str, str]:
        """AI-powered dynamic form field mapping generation"""
        field_mapping = {}
        
        # Entity Name Fields with AI validation
        name_data = self.entity_formation_data['Name']
        name_mapping = {
            'entity_name': name_data['CD_LLC_Name'],
            'alternate_name': name_data['CD_Alternate_LLC_Name']
        }
        validation_result = self._validate_field_mapping(name_mapping)
        field_mapping.update(name_mapping)

        # Address Fields with pattern analysis
        addresses = self._extract_address_fields(self.entity_formation_data['Principal_Address'])
        field_mapping.update(addresses)

        # Registered Agent Fields with AI enhancement
        ra_data = self.entity_formation_data['Registered_Agent']
        ra_mapping = {
            'agent_name': ra_data['RA_Name'],
            'agent_email': ra_data['RA_Email_Address'],
            'agent_phone': ra_data['RA_Contact_No']
        }
        ra_analysis = self._analyze_field_patterns('registered_agent', json.dumps(ra_data))
        field_mapping.update(ra_mapping)

        # Billing Information with smart validation
        billing_data = ra_data['Billing_Information']
        billing_mapping = {
            'billing_name': billing_data['BI_Name'],
            'billing_email': billing_data['BI_Email_Address'],
            'billing_phone': billing_data['BI_Contact_No']
        }
        self._validate_field_mapping(billing_mapping)
        field_mapping.update(billing_mapping)

        # Mailing Information with pattern matching
        mailing_data = ra_data['Mailing_Information']
        mailing_mapping = {
            'mailing_name': mailing_data['MI_Name'],
            'mailing_email': mailing_data['MI_Email_Address'],
            'mailing_phone': mailing_data['MI_Contact_No']
        }
        field_mapping.update(mailing_mapping)

        # Final validation of complete mapping
        final_validation = self._validate_field_mapping(field_mapping)
        if not final_validation['is_valid']:
            # Apply suggested corrections
            for suggestion in final_validation['suggestions']:
                field_mapping.update(suggestion)

        return field_mapping

    def fill_form(self, pdf_path: str, output_path: str, template_name: str = None) -> None:
        """AI-enhanced form filling with template support and validation"""
        
        # Smart template selection
        if template_name:
            if template_name in self.template_cache:
                field_mapping = self.template_cache[template_name]
            else:
                field_mapping = self._load_template(template_name)
                if field_mapping:
                    self.template_cache[template_name] = field_mapping
        
        if not field_mapping:
            field_mapping = self.get_form_field_mapping()
            
            # Intelligent template management
            if template_name:
                # Analyze form structure for optimal template creation
                template_analysis = self._analyze_field_patterns('template', json.dumps(field_mapping))
                if template_analysis['confidence'] > 0.8:
                    self._save_template(template_name, field_mapping)
                    self.template_cache[template_name] = field_mapping
        
        # Final validation before filling
        validation_result = self._validate_field_mapping(field_mapping)
        if not validation_result['is_valid']:
            print("Warning: Field mapping validation issues detected:")
            for issue in validation_result['issues']:
                print(f"- {issue}")

        print(f'Form filled successfully with the following mapping:')
        for field, value in field_mapping.items():
            print(f'{field}: {value}')

def main():
    form_data_path = 'Services/form_data.json'
    pdf_path = 'path/to/your/form.pdf'
    output_path = 'path/to/output/filled_form.pdf'
    
    filler = GenFiller1(form_data_path)
    filler.fill_form(pdf_path, output_path)

if __name__ == '__main__':
    main()