from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *

class TemplateAnalysisResult(BaseModel):
    template_id: str
    form_type: str
    fields: List[Dict[str, Any]]
    validation_rules: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

class TemplateManager:
    def __init__(self, api_key: str):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt="""
            You are an expert AI system for PDF form template analysis and management.
            Analyze form structures, identify field patterns, and generate validation rules.
            Learn from previous form analyses to improve field matching and validation.
            """
        )
        self.template_cache = {}
        self.template_store_path = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(self.template_store_path, exist_ok=True)
    
    async def analyze_template(self, pdf_path: str) -> TemplateAnalysisResult:
        """Analyze a PDF form template and extract field patterns and validation rules"""
        try:
            analysis_prompt = f"""
            Analyze this PDF form template and provide:
            1. Form type classification
            2. Field identification and patterns
            3. Validation rules for each field
            4. Common field groups and relationships
            5. Required vs optional fields
            6. Field format requirements
            
            Return the analysis as a structured JSON object.
            """
            
            response = await self.agent.run(analysis_prompt)
            analysis_data = json.loads(response.data)
            
            template_id = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = TemplateAnalysisResult(
                template_id=template_id,
                form_type=analysis_data['form_type'],
                fields=analysis_data['fields'],
                validation_rules=analysis_data['validation_rules'],
                metadata={
                    'source_pdf': os.path.basename(pdf_path),
                    'analysis_version': '1.0',
                    'field_count': len(analysis_data['fields'])
                },
                created_at=datetime.now()
            )
            
            # Save template analysis
            self._save_template(result)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing template: {str(e)}")
            raise
    
    def _save_template(self, template: TemplateAnalysisResult):
        """Save template analysis to persistent storage"""
        template_path = os.path.join(
            self.template_store_path,
            f"{template.template_id}.json"
        )
        
        with open(template_path, 'w') as f:
            json.dump(template.model_dump(), f, indent=2, default=str)
        
        # Update cache
        self.template_cache[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[TemplateAnalysisResult]:
        """Retrieve a template analysis by ID"""
        if template_id in self.template_cache:
            return self.template_cache[template_id]
            
        template_path = os.path.join(self.template_store_path, f"{template_id}.json")
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                data = json.load(f)
                template = TemplateAnalysisResult(**data)
                self.template_cache[template_id] = template
                return template
                
        return None
    
    async def validate_form_data(self, template_id: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate form data against template rules"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
            
        validation_prompt = f"""
        Validate this form data against the template rules:
        Template Rules: {json.dumps(template.validation_rules, indent=2)}
        Form Data: {json.dumps(form_data, indent=2)}
        
        Return a JSON object with validation results and any errors found.
        """
        
        response = await self.agent.run(validation_prompt)
        return json.loads(response.data)
    
    async def suggest_field_mapping(self, template_id: str, field_name: str) -> Dict[str, Any]:
        """Suggest field mappings based on template analysis"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
            
        mapping_prompt = f"""
        Suggest mappings for this field based on the template analysis:
        Field Name: {field_name}
        Template Fields: {json.dumps(template.fields, indent=2)}
        
        Consider:
        1. Field name similarity
        2. Field type compatibility
        3. Common patterns and relationships
        4. Previous successful mappings
        
        Return a JSON object with mapping suggestions and confidence scores.
        """
        
        response = await self.agent.run(mapping_prompt)
        return json.loads(response.data)
