import asyncio
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from imageextractor.Common.constants import API_KEY


class RiskLevel(str, Enum):
    """Risk levels for medical conditions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class ConditionCategory(str, Enum):
    """Categories of medical conditions."""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    GASTROINTESTINAL = "gastrointestinal"
    ENDOCRINE = "endocrine"
    IMMUNOLOGICAL = "immunological"
    ONCOLOGICAL = "oncological"
    PSYCHIATRIC = "psychiatric"
    ORTHOPEDIC = "orthopedic"
    DERMATOLOGICAL = "dermatological"


class TreatmentType(str, Enum):
    """Types of medical treatments."""
    MEDICATION = "medication"
    SURGERY = "surgery"
    THERAPY = "therapy"
    LIFESTYLE = "lifestyle"
    PREVENTIVE = "preventive"
    EMERGENCY = "emergency"
    PALLIATIVE = "palliative"


class Symptom(BaseModel):
    """Model for patient symptoms."""
    symptom_id: str = Field(..., description="Unique identifier for the symptom")
    name: str = Field(..., description="Name of the symptom")
    severity: int = Field(..., ge=1, le=10, description="Severity level (1-10)")
    duration: str = Field(..., description="Duration of the symptom")
    frequency: str = Field(..., description="Frequency of occurrence")
    triggers: List[str] = Field(default_factory=list, description="Known triggers")
    associated_symptoms: List[str] = Field(default_factory=list, description="Related symptoms")
    
    @field_validator('severity')
    def validate_severity(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Severity must be between 1 and 10")
        return v


class MedicalCondition(BaseModel):
    """Model for medical conditions."""
    condition_id: str = Field(..., description="Unique identifier for the condition")
    name: str = Field(..., description="Name of the medical condition")
    category: ConditionCategory = Field(..., description="Category of the condition")
    risk_level: RiskLevel = Field(..., description="Risk level of the condition")
    description: str = Field(..., description="Detailed description")
    symptoms: List[str] = Field(default_factory=list, description="Common symptoms")
    causes: List[str] = Field(default_factory=list, description="Known causes")
    complications: List[str] = Field(default_factory=list, description="Potential complications")
    icd_code: Optional[str] = Field(None, description="ICD-10 code if applicable")
    prevalence: Optional[str] = Field(None, description="Prevalence information")


class TreatmentRecommendation(BaseModel):
    """Model for treatment recommendations."""
    treatment_id: str = Field(..., description="Unique identifier for the treatment")
    type: TreatmentType = Field(..., description="Type of treatment")
    name: str = Field(..., description="Name of the treatment")
    description: str = Field(..., description="Detailed description")
    effectiveness: float = Field(..., ge=0, le=100, description="Effectiveness percentage")
    side_effects: List[str] = Field(default_factory=list, description="Potential side effects")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")
    duration: str = Field(..., description="Expected treatment duration")
    cost_estimate: Optional[str] = Field(None, description="Estimated cost")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1-5)")
    
    @field_validator('effectiveness')
    def validate_effectiveness(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Effectiveness must be between 0 and 100")
        return v


class PatientProfile(BaseModel):
    """Model for patient profile."""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., description="Patient gender")
    weight: Optional[float] = Field(None, ge=0, description="Weight in kg")
    height: Optional[float] = Field(None, ge=0, description="Height in cm")
    medical_history: List[str] = Field(default_factory=list, description="Medical history")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    lifestyle_factors: Dict[str, Any] = Field(default_factory=dict, description="Lifestyle information")
    family_history: List[str] = Field(default_factory=list, description="Family medical history")
    
    @field_validator('age')
    def validate_age(cls, v):
        if not 0 <= v <= 150:
            raise ValueError("Age must be between 0 and 150")
        return v


class DiagnosisAnalysis(BaseModel):
    """Model for comprehensive diagnosis analysis."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    patient_profile: PatientProfile = Field(..., description="Patient information")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    primary_symptoms: List[Symptom] = Field(default_factory=list)
    differential_diagnosis: List[MedicalCondition] = Field(default_factory=list)
    primary_diagnosis: Optional[MedicalCondition] = Field(None, description="Primary diagnosis")
    confidence_score: float = Field(..., ge=0, le=100, description="Diagnosis confidence (0-100)")
    risk_assessment: RiskLevel = Field(..., description="Overall risk assessment")
    treatment_recommendations: List[TreatmentRecommendation] = Field(default_factory=list)
    preventive_measures: List[str] = Field(default_factory=list)
    follow_up_plan: str = Field(..., description="Recommended follow-up plan")
    urgency_level: int = Field(..., ge=1, le=5, description="Urgency level (1-5)")
    specialist_referral: Optional[str] = Field(None, description="Recommended specialist")
    
    @field_validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Confidence score must be between 0 and 100")
        return v


class AnalysisModule(ABC):
    """Abstract base class for medical analysis modules."""
    
    @abstractmethod
    async def analyze(self, patient_data: Dict[str, Any], symptoms: List[Symptom], **kwargs) -> Dict[str, Any]:
        """Perform medical analysis and return results."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this analysis module."""
        pass


class SymptomAnalysisModule(AnalysisModule):
    """Symptom analysis and pattern recognition module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert medical symptom analyst specializing in:
            - Symptom pattern recognition
            - Differential diagnosis
            - Symptom severity assessment
            - Risk factor analysis
            - Symptom correlation analysis
            - Red flag identification
            """,
            retries=2,
        )
    
    async def analyze(self, patient_data: Dict[str, Any], symptoms: List[Symptom], **kwargs) -> Dict[str, Any]:
        age = patient_data.get('age', 'Unknown')
        gender = patient_data.get('gender', 'Unknown')
        medical_history = patient_data.get('medical_history', [])
        
        symptoms_text = "\n".join([f"- {s.name}: Severity {s.severity}/10, Duration: {s.duration}" for s in symptoms])
        
        prompt = f"""
        Analyze the following patient symptoms for medical diagnosis:
        
        Patient Information:
        - Age: {age}
        - Gender: {gender}
        - Medical History: {', '.join(medical_history) if medical_history else 'None'}
        
        Symptoms:
        {symptoms_text}
        
        Provide comprehensive symptom analysis in JSON format with:
        - symptom_patterns (identified symptom patterns)
        - differential_diagnosis (possible conditions)
        - risk_factors (identified risk factors)
        - red_flags (concerning symptoms)
        - urgency_assessment (urgency level 1-5)
        - specialist_recommendations (specialist types needed)
        - diagnostic_tests (recommended tests)
        - symptom_correlations (how symptoms relate)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Symptom analysis failed: {str(e)}"}
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'data'):
            return response.data
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def get_name(self) -> str:
        return "symptom_analysis"


class TreatmentRecommendationModule(AnalysisModule):
    """Treatment recommendation and planning module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert medical treatment planner specializing in:
            - Evidence-based treatment recommendations
            - Medication management
            - Treatment effectiveness assessment
            - Side effect analysis
            - Contraindication checking
            - Cost-benefit analysis
            - Treatment prioritization
            """,
            retries=2,
        )
    
    async def analyze(self, patient_data: Dict[str, Any], symptoms: List[Symptom], **kwargs) -> Dict[str, Any]:
        diagnosis = kwargs.get('diagnosis', 'Unknown')
        allergies = patient_data.get('allergies', [])
        current_medications = patient_data.get('medications', [])
        
        symptoms_text = "\n".join([f"- {s.name}: Severity {s.severity}/10" for s in symptoms])
        
        prompt = f"""
        Provide treatment recommendations for the following case:
        
        Diagnosis: {diagnosis}
        Patient Allergies: {', '.join(allergies) if allergies else 'None'}
        Current Medications: {', '.join(current_medications) if current_medications else 'None'}
        
        Symptoms:
        {symptoms_text}
        
        Provide treatment recommendations in JSON format with:
        - medication_recommendations (specific medications)
        - lifestyle_recommendations (lifestyle changes)
        - therapy_recommendations (therapeutic interventions)
        - preventive_measures (prevention strategies)
        - follow_up_plan (monitoring plan)
        - specialist_referrals (specialist recommendations)
        - treatment_priorities (priority order)
        - cost_estimates (treatment costs)
        - effectiveness_rates (expected outcomes)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Treatment recommendation failed: {str(e)}"}
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'data'):
            return response.data
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def get_name(self) -> str:
        return "treatment_recommendation"


class RiskAssessmentModule(AnalysisModule):
    """Risk assessment and prognosis module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert medical risk assessor specializing in:
            - Risk factor analysis
            - Prognosis assessment
            - Complication prediction
            - Emergency risk evaluation
            - Long-term outcome analysis
            - Risk stratification
            - Preventive risk management
            """,
            retries=2,
        )
    
    async def analyze(self, patient_data: Dict[str, Any], symptoms: List[Symptom], **kwargs) -> Dict[str, Any]:
        age = patient_data.get('age', 'Unknown')
        medical_history = patient_data.get('medical_history', [])
        family_history = patient_data.get('family_history', [])
        
        symptoms_text = "\n".join([f"- {s.name}: Severity {s.severity}/10" for s in symptoms])
        
        prompt = f"""
        Assess medical risks for the following patient:
        
        Patient Age: {age}
        Medical History: {', '.join(medical_history) if medical_history else 'None'}
        Family History: {', '.join(family_history) if family_history else 'None'}
        
        Current Symptoms:
        {symptoms_text}
        
        Provide risk assessment in JSON format with:
        - risk_factors (identified risk factors)
        - emergency_risks (immediate risks)
        - long_term_risks (long-term complications)
        - prognosis_assessment (expected outcomes)
        - risk_stratification (risk levels)
        - preventive_measures (risk prevention)
        - monitoring_requirements (monitoring needs)
        - emergency_indicators (warning signs)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'data'):
            return response.data
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def get_name(self) -> str:
        return "risk_assessment"


class PreventiveCareModule(AnalysisModule):
    """Preventive care and wellness planning module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert preventive care specialist specializing in:
            - Preventive screening recommendations
            - Vaccination schedules
            - Lifestyle optimization
            - Health maintenance planning
            - Early detection strategies
            - Wellness coaching
            - Preventive interventions
            """,
            retries=2,
        )
    
    async def analyze(self, patient_data: Dict[str, Any], symptoms: List[Symptom], **kwargs) -> Dict[str, Any]:
        age = patient_data.get('age', 'Unknown')
        gender = patient_data.get('gender', 'Unknown')
        lifestyle_factors = patient_data.get('lifestyle_factors', {})
        
        prompt = f"""
        Provide preventive care recommendations for:
        
        Patient Age: {age}
        Gender: {gender}
        Lifestyle Factors: {lifestyle_factors}
        
        Provide preventive care analysis in JSON format with:
        - screening_recommendations (preventive screenings)
        - vaccination_schedule (vaccination needs)
        - lifestyle_recommendations (lifestyle improvements)
        - wellness_plan (overall wellness strategy)
        - early_detection_measures (early detection)
        - health_maintenance (maintenance activities)
        - preventive_interventions (preventive measures)
        - wellness_goals (health goals)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Preventive care analysis failed: {str(e)}"}
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'data'):
            return response.data
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def get_name(self) -> str:
        return "preventive_care"


class IntelligentHealthcareSystem:
    """Intelligent Healthcare Diagnosis and Treatment Recommendation System."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", 
                 enabled_modules: List[str] = None):
        """
        Initialize the Intelligent Healthcare System.
        
        Args:
            api_key: API key for the LLM model
            model: Model name to use
            enabled_modules: List of module names to enable. If None, all modules are enabled.
        """
        self.llm = GeminiModel(model, provider=GoogleGLAProvider(api_key=api_key))
        self.api_key = api_key
        self.analysis_history = []
        self.enabled_modules = enabled_modules or self._get_default_modules()
        
        # Initialize modules
        self.modules = self._initialize_modules()
    
    def _get_default_modules(self) -> List[str]:
        """Get list of default modules to enable."""
        return [
            "symptom_analysis",
            "treatment_recommendation",
            "risk_assessment",
            "preventive_care"
        ]
    
    def _initialize_modules(self) -> Dict[str, AnalysisModule]:
        """Initialize analysis modules based on configuration."""
        module_classes = {
            "symptom_analysis": SymptomAnalysisModule,
            "treatment_recommendation": TreatmentRecommendationModule,
            "risk_assessment": RiskAssessmentModule,
            "preventive_care": PreventiveCareModule
        }
        
        modules = {}
        for module_name in self.enabled_modules:
            if module_name in module_classes:
                modules[module_name] = module_classes[module_name](self.llm)
        
        return modules
    
    async def analyze_patient(self, patient_profile: PatientProfile, symptoms: List[Symptom], 
                             **kwargs) -> DiagnosisAnalysis:
        """Analyze patient symptoms and provide comprehensive diagnosis."""
        
        print(f"\n=== Starting Intelligent Healthcare Analysis ===")
        print(f"Patient ID: {patient_profile.patient_id}")
        print(f"Age: {patient_profile.age}, Gender: {patient_profile.gender}")
        print(f"Enabled modules: {', '.join(self.enabled_modules)}")
        
        # Convert patient profile to dict for analysis
        patient_data = patient_profile.model_dump()
        
        # Run all enabled modules
        module_results = {}
        
        for module_name, module in self.modules.items():
            print(f"Running {module_name} analysis...")
            
            module_kwargs = kwargs.copy()
            module_result = await module.analyze(patient_data, symptoms, **module_kwargs)
            module_results[module_name] = module_result
        
        # Compile comprehensive analysis
        analysis = await self._compile_analysis(patient_profile, symptoms, module_results, **kwargs)
        
        # Store analysis history
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_profile.patient_id,
            "enabled_modules": self.enabled_modules,
            "results": analysis.model_dump()
        })
        
        return analysis
    
    async def _compile_analysis(self, patient_profile: PatientProfile, symptoms: List[Symptom],
                               module_results: Dict[str, Any], **kwargs) -> DiagnosisAnalysis:
        """Compile results from all modules into a comprehensive diagnosis."""
        
        # Generate unique analysis ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract differential diagnosis from symptom analysis
        differential_diagnosis = []
        primary_diagnosis = None
        confidence_score = 0.0
        
        if "symptom_analysis" in module_results:
            symptom_result = module_results["symptom_analysis"]
            if isinstance(symptom_result, dict) and "differential_diagnosis" in symptom_result:
                for condition_data in symptom_result["differential_diagnosis"]:
                    condition = MedicalCondition(
                        condition_id=f"condition_{len(differential_diagnosis)}",
                        name=condition_data.get("name", "Unknown Condition"),
                        category=ConditionCategory.NEUROLOGICAL,  # Default category
                        risk_level=RiskLevel.MEDIUM,
                        description=condition_data.get("description", ""),
                        symptoms=condition_data.get("symptoms", []),
                        causes=condition_data.get("causes", []),
                        complications=condition_data.get("complications", [])
                    )
                    differential_diagnosis.append(condition)
                
                # Set primary diagnosis as the first one (most likely)
                if differential_diagnosis:
                    primary_diagnosis = differential_diagnosis[0]
                    confidence_score = symptom_result.get("confidence_score", 75.0)
        
        # Extract treatment recommendations
        treatment_recommendations = []
        
        if "treatment_recommendation" in module_results:
            treatment_result = module_results["treatment_recommendation"]
            if isinstance(treatment_result, dict) and "medication_recommendations" in treatment_result:
                for i, med_data in enumerate(treatment_result["medication_recommendations"]):
                    treatment = TreatmentRecommendation(
                        treatment_id=f"treatment_{i}",
                        type=TreatmentType.MEDICATION,
                        name=med_data.get("name", "Unknown Treatment"),
                        description=med_data.get("description", ""),
                        effectiveness=med_data.get("effectiveness", 80.0),
                        side_effects=med_data.get("side_effects", []),
                        contraindications=med_data.get("contraindications", []),
                        duration=med_data.get("duration", "Unknown"),
                        priority=med_data.get("priority", 3)
                    )
                    treatment_recommendations.append(treatment)
        
        # Determine risk assessment
        risk_assessment = RiskLevel.MEDIUM
        urgency_level = 3
        
        if "risk_assessment" in module_results:
            risk_result = module_results["risk_assessment"]
            if isinstance(risk_result, dict):
                risk_level = risk_result.get("risk_level", "medium")
                risk_assessment = RiskLevel(risk_level.upper())
                urgency_level = risk_result.get("urgency_level", 3)
        
        # Generate preventive measures
        preventive_measures = []
        
        if "preventive_care" in module_results:
            preventive_result = module_results["preventive_care"]
            if isinstance(preventive_result, dict):
                preventive_measures = preventive_result.get("lifestyle_recommendations", [])
        
        # Determine follow-up plan
        follow_up_plan = "Schedule follow-up appointment in 2 weeks"
        if urgency_level >= 4:
            follow_up_plan = "Immediate follow-up required"
        elif urgency_level <= 2:
            follow_up_plan = "Routine follow-up in 1 month"
        
        # Determine specialist referral
        specialist_referral = None
        if "symptom_analysis" in module_results:
            symptom_result = module_results["symptom_analysis"]
            if isinstance(symptom_result, dict):
                specialists = symptom_result.get("specialist_recommendations", [])
                if specialists:
                    specialist_referral = specialists[0]
        
        return DiagnosisAnalysis(
            analysis_id=analysis_id,
            patient_profile=patient_profile,
            primary_symptoms=symptoms,
            differential_diagnosis=differential_diagnosis,
            primary_diagnosis=primary_diagnosis,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            treatment_recommendations=treatment_recommendations,
            preventive_measures=preventive_measures,
            follow_up_plan=follow_up_plan,
            urgency_level=urgency_level,
            specialist_referral=specialist_referral
        )
    
    def print_analysis_results(self, analysis: DiagnosisAnalysis):
        """Print comprehensive analysis results."""
        
        print(f"\n=== Healthcare Analysis Results ===")
        print(f"Analysis ID: {analysis.analysis_id}")
        print(f"Patient ID: {analysis.patient_profile.patient_id}")
        print(f"Analysis Date: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Confidence Score: {analysis.confidence_score:.1f}%")
        print(f"Risk Assessment: {analysis.risk_assessment.value.upper()}")
        print(f"Urgency Level: {analysis.urgency_level}/5")
        print(f"Follow-up Plan: {analysis.follow_up_plan}")
        
        if analysis.specialist_referral:
            print(f"Specialist Referral: {analysis.specialist_referral}")
        
        # Print symptoms
        if analysis.primary_symptoms:
            print(f"\n📋 Primary Symptoms ({len(analysis.primary_symptoms)}):")
            for i, symptom in enumerate(analysis.primary_symptoms, 1):
                print(f"  {i}. {symptom.name} (Severity: {symptom.severity}/10)")
                print(f"     Duration: {symptom.duration}")
                print(f"     Frequency: {symptom.frequency}")
        
        # Print differential diagnosis
        if analysis.differential_diagnosis:
            print(f"\n🔍 Differential Diagnosis ({len(analysis.differential_diagnosis)}):")
            for i, condition in enumerate(analysis.differential_diagnosis, 1):
                print(f"  {i}. {condition.name}")
                print(f"     Category: {condition.category.value}")
                print(f"     Risk Level: {condition.risk_level.value.upper()}")
                print(f"     Description: {condition.description}")
        
        # Print treatment recommendations
        if analysis.treatment_recommendations:
            print(f"\n💊 Treatment Recommendations ({len(analysis.treatment_recommendations)}):")
            for i, treatment in enumerate(analysis.treatment_recommendations, 1):
                print(f"  {i}. {treatment.name}")
                print(f"     Type: {treatment.type.value}")
                print(f"     Effectiveness: {treatment.effectiveness:.1f}%")
                print(f"     Priority: {treatment.priority}/5")
                print(f"     Duration: {treatment.duration}")
        
        # Print preventive measures
        if analysis.preventive_measures:
            print(f"\n🛡️ Preventive Measures ({len(analysis.preventive_measures)}):")
            for i, measure in enumerate(analysis.preventive_measures, 1):
                print(f"  {i}. {measure}")
        
        print(f"\n=== Analysis Complete ===")
    
    def export_analysis(self, analysis: DiagnosisAnalysis, format: str = "json") -> str:
        """Export analysis results in specified format."""
        if format.lower() == "json":
            return analysis.model_dump_json(indent=2)
        elif format.lower() == "csv":
            # Convert analysis to CSV format
            csv_data = []
            
            # Add main analysis data
            csv_data.append(["Analysis ID", analysis.analysis_id])
            csv_data.append(["Patient ID", analysis.patient_profile.patient_id])
            csv_data.append(["Confidence Score", f"{analysis.confidence_score:.1f}%"])
            csv_data.append(["Risk Assessment", analysis.risk_assessment.value])
            csv_data.append(["Urgency Level", f"{analysis.urgency_level}/5"])
            csv_data.append(["Follow-up Plan", analysis.follow_up_plan])
            
            # Add symptoms
            for symptom in analysis.primary_symptoms:
                csv_data.append([
                    "Symptom",
                    symptom.name,
                    f"{symptom.severity}/10",
                    symptom.duration,
                    symptom.frequency
                ])
            
            # Add conditions
            for condition in analysis.differential_diagnosis:
                csv_data.append([
                    "Condition",
                    condition.name,
                    condition.category.value,
                    condition.risk_level.value,
                    condition.description
                ])
            
            # Add treatments
            for treatment in analysis.treatment_recommendations:
                csv_data.append([
                    "Treatment",
                    treatment.name,
                    treatment.type.value,
                    f"{treatment.effectiveness:.1f}%",
                    f"{treatment.priority}/5",
                    treatment.duration
                ])
            
            df = pd.DataFrame(csv_data, columns=["Type", "Name", "Severity/Risk", "Category/Type", "Description", "Additional Info"])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history
    
    def add_module(self, module: AnalysisModule):
        """Add a custom analysis module."""
        self.modules[module.get_name()] = module
        self.enabled_modules.append(module.get_name())
    
    def remove_module(self, module_name: str):
        """Remove an analysis module."""
        if module_name in self.modules:
            del self.modules[module_name]
            if module_name in self.enabled_modules:
                self.enabled_modules.remove(module_name)


async def main():
    """Main function to demonstrate the Intelligent Healthcare System."""
    api_key = API_KEY
    
    # Configuration
    config = {
        "patient_profile": {
            "patient_id": "P12345",
            "age": 45,
            "gender": "Female",
            "weight": 65.0,
            "height": 165.0,
            "medical_history": ["Hypertension", "Diabetes Type 2"],
            "allergies": ["Penicillin"],
            "medications": ["Metformin", "Lisinopril"],
            "lifestyle_factors": {
                "smoking": False,
                "exercise": "Moderate",
                "diet": "Balanced"
            },
            "family_history": ["Heart Disease", "Diabetes"]
        },
        "symptoms": [
            {
                "symptom_id": "S001",
                "name": "Chest Pain",
                "severity": 7,
                "duration": "2 hours",
                "frequency": "First time",
                "triggers": ["Physical exertion"],
                "associated_symptoms": ["Shortness of breath", "Sweating"]
            },
            {
                "symptom_id": "S002",
                "name": "Shortness of Breath",
                "severity": 6,
                "duration": "1 hour",
                "frequency": "Occasional",
                "triggers": ["Climbing stairs"],
                "associated_symptoms": ["Fatigue"]
            }
        ],
        "enabled_modules": [
            "symptom_analysis",
            "treatment_recommendation",
            "risk_assessment",
            "preventive_care"
        ]
    }
    
    # Initialize healthcare system
    healthcare_system = IntelligentHealthcareSystem(
        api_key=api_key,
        enabled_modules=config["enabled_modules"]
    )
    
    try:
        # Create patient profile
        patient_profile = PatientProfile(**config["patient_profile"])
        
        # Create symptoms
        symptoms = [Symptom(**symptom_data) for symptom_data in config["symptoms"]]
        
        # Run analysis
        analysis = await healthcare_system.analyze_patient(
            patient_profile=patient_profile,
            symptoms=symptoms
        )
        
        # Print results
        healthcare_system.print_analysis_results(analysis)
        
        # Export results
        json_export = healthcare_system.export_analysis(analysis, "json")
        print(f"\nJSON Export (first 500 chars):\n{json_export[:500]}...")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 