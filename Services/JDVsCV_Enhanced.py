import fitz
import asyncio
import os
import json
import cv2
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
# import speech_recognition as sr  # Commented out as it's not essential
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.tools import ToolDefinition
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod

from pydantic_ai.providers.google_gla import GoogleGLAProvider


from imageextractor.Common.constants import *


class AnalysisModule(ABC):
    """Abstract base class for analysis modules."""
    
    @abstractmethod
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        """Perform analysis and return results."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this analysis module."""
        pass


class BasicMatchingModule(AnalysisModule):
    """Core CV-JD matching analysis."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert recruiter AI analyzing resumes and job descriptions.
            Your task is to:
            - Extract and match the relevant skills, qualifications, and experiences from the CV and JD.
            - Exclude any irrelevant information such as POS tags, common words, and other non-skill-related content.
            - Identify missing skills or gaps in the CV based on JD requirements.
            - Provide a match score (0-100%) and a comprehensive assessment.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        Compare the following CV and JD and provide the following details:
        - Match Score (percentage) as match_score
        - Matching Skills (relevant skills only) as matching_skills
        - Missing Requirements (relevant skills only) as missing_requirements
        - Match Category (Best Match / Good Match / Average Match / Poor Match) as match_category
        - Overall Assessment as overall_assessment

        Exclude any irrelevant information such as POS tags, common words, and other non-skill-related content and return all of this in json format.

        CV Text:
        {cv_text}

        JD Text:
        {jd_text}
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Basic matching analysis failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "basic_matching"


class CultureFitModule(AnalysisModule):
    """Culture fit and personality analysis."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert in organizational psychology and culture analysis.
            Analyze communication styles, values, and cultural fit between candidates and organizations.
            Focus on soft skills, work style preferences, and cultural alignment.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        company_values = kwargs.get('company_values', '')
        
        prompt = f"""
        Analyze the cultural fit between the candidate and the company based on:
        
        CV Text: {cv_text}
        Company Values: {company_values}
        
        Provide analysis in JSON format with:
        - culture_fit_score (0-100)
        - communication_style_match
        - work_style_alignment
        - values_compatibility
        - team_collaboration_potential
        - cultural_risks
        - overall_culture_assessment
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Culture fit analysis failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "culture_fit"


class CareerTrajectoryModule(AnalysisModule):
    """Career trajectory and growth prediction."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert career counselor and industry analyst.
            Analyze career progression patterns, industry trends, and predict future skill requirements.
            Provide insights on career trajectory and growth potential.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        industry_trends = kwargs.get('industry_trends', 'Technology industry trends')
        
        prompt = f"""
        Analyze the career trajectory and predict future requirements:
        
        CV Text: {cv_text}
        Industry Trends: {industry_trends}
        
        Provide analysis in JSON format with:
        - career_growth_potential (0-100)
        - predicted_next_roles
        - skill_evolution_timeline
        - industry_relevance_score
        - future_skill_requirements
        - career_risk_factors
        - growth_recommendations
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Career trajectory analysis failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "career_trajectory"


class ResumeOptimizationModule(AnalysisModule):
    """Resume optimization for specific job descriptions."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert resume writer and career coach.
            Optimize resumes for specific job descriptions by improving keywords, structure, and content.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        Optimize the following resume for the specific job description:
        
        Original CV: {cv_text}
        Job Description: {jd_text}
        
        Provide optimization recommendations in JSON format with:
        - keyword_optimization
        - section_reordering
        - content_enhancements
        - skill_highlighting
        - experience_reframing
        - overall_score_improvement
        - specific_edits
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Resume optimization failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "resume_optimization"


class InterviewPreparationModule(AnalysisModule):
    """Interview question generation and preparation."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert interview coach and HR professional.
            Generate personalized interview questions, technical assessments, and behavioral scenarios.
            Focus on role-specific and candidate-specific questions.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        prompt = f"""
        Generate personalized interview questions based on:
        
        CV Text: {cv_text}
        Job Description: {jd_text}
        
        Provide comprehensive interview preparation in JSON format with:
        - technical_questions
        - behavioral_questions
        - situational_questions
        - skill_assessment_tasks
        - culture_fit_questions
        - problem_solving_scenarios
        - follow_up_questions
        - evaluation_criteria
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Interview preparation failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "interview_preparation"


class BiasDetectionModule(AnalysisModule):
    """Bias detection and fairness analysis."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert in fair hiring practices and bias detection.
            Analyze recruitment processes for potential biases and provide mitigation strategies.
            """,
            retries=2,
        )
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        analysis_result = kwargs.get('analysis_result', {})
        
        # Convert analysis_result to string to avoid JSON serialization issues
        analysis_str = str(analysis_result) if analysis_result else "{}"
        
        prompt = f"""
        Analyze the following recruitment analysis for potential biases:
        
        Analysis Result: {analysis_str}
        
        Identify potential biases in JSON format with:
        - gender_bias_indicators
        - age_bias_indicators
        - cultural_bias_indicators
        - educational_bias_indicators
        - bias_mitigation_recommendations
        - fairness_score
        - bias_explanation
        """
        
        try:
            response = await self.agent.run(prompt)
            # Extract the actual data from the response
            if hasattr(response, 'output'):
                return response.output
            elif hasattr(response, 'data'):
                return response.data
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            return {"error": f"Bias detection failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "bias_detection"


class MarketIntelligenceModule(AnalysisModule):
    """Market intelligence and salary analysis."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.market_data_sources = {}
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        role = kwargs.get('role', 'Software Engineer')
        location = kwargs.get('location', 'Remote')
        
        # In a real implementation, this would connect to external APIs
        market_data = await self._get_market_data(role, location)
        
        return market_data
    
    async def _get_market_data(self, role: str, location: str) -> Dict[str, Any]:
        """Get market data from various sources."""
        # Simulated market data - replace with real API calls
        return {
            "salary_range": {
                "min": 60000,
                "max": 120000,
                "median": 85000
            },
            "skill_demand": {
                "python": 85,
                "javascript": 78,
                "react": 72,
                "aws": 68,
                "docker": 65
            },
            "market_trends": {
                "growing_demand": True,
                "skill_evolution": "Rapid",
                "remote_work_adoption": 75
            },
            "competitive_landscape": {
                "candidate_supply": "Medium",
                "hiring_difficulty": "Moderate",
                "time_to_fill": "4-6 weeks"
            }
        }
    
    def get_name(self) -> str:
        return "market_intelligence"


class VisualAnalysisModule(AnalysisModule):
    """Visual analysis of resumes and portfolios."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        image_path = kwargs.get('image_path')
        
        if not image_path:
            return {"error": "No image path provided for visual analysis"}
        
        return await self._analyze_visual_elements(image_path)
    
    async def _analyze_visual_elements(self, image_path: str) -> Dict[str, Any]:
        """Analyze visual elements in resumes and portfolios."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            # Edge detection for structure analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            return {
                "image_dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "color_profile": {
                    "avg_saturation": round(avg_saturation, 2),
                    "avg_brightness": round(avg_value, 2)
                },
                "structure_analysis": {
                    "edge_density": round(edge_density, 4),
                    "complexity_score": round(edge_density * 100, 2)
                },
                "design_indicators": {
                    "professional_layout": edge_density > 0.01,
                    "color_balance": avg_saturation < 100,
                    "visual_clarity": avg_value > 100
                }
            }
            
        except Exception as e:
            return {"error": f"Visual analysis failed: {str(e)}"}
    
    def get_name(self) -> str:
        return "visual_analysis"


class EnhancedCVJDMatcher:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", 
                 enabled_modules: List[str] = None):
        """
        Enhanced JD-CV Matcher with modular architecture.
        
        Args:
            api_key: API key for the LLM model
            model: Model name to use
            enabled_modules: List of module names to enable. If None, all modules are enabled.
        """
        self.llm = GeminiModel(model, provider=GoogleGLAProvider(api_key=API_KEY))
        self.api_key = api_key
        self.analysis_history = []
        self.enabled_modules = enabled_modules or self._get_default_modules()
        
        # Initialize modules
        self.modules = self._initialize_modules()
    
    def _get_default_modules(self) -> List[str]:
        """Get list of default modules to enable."""
        return [
            "basic_matching",
            "culture_fit", 
            "career_trajectory",
            "resume_optimization",
            "interview_preparation",
            "bias_detection",
            "market_intelligence"
        ]
    
    def _initialize_modules(self) -> Dict[str, AnalysisModule]:
        """Initialize analysis modules based on configuration."""
        module_classes = {
            "basic_matching": BasicMatchingModule,
            "culture_fit": CultureFitModule,
            "career_trajectory": CareerTrajectoryModule,
            "resume_optimization": ResumeOptimizationModule,
            "interview_preparation": InterviewPreparationModule,
            "bias_detection": BiasDetectionModule,
            "market_intelligence": MarketIntelligenceModule,
            "visual_analysis": VisualAnalysisModule
        }
        
        modules = {}
        for module_name in self.enabled_modules:
            if module_name in module_classes:
                modules[module_name] = module_classes[module_name](self.llm)
        
        return modules
    
    async def extract_pdf_text_tool(self, ctx: RunContext, file_path: str) -> str:
        """Enhanced PDF text extraction with better error handling."""
        extracted_text = []
        try:
            with fitz.open(file_path) as pdf:
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    text = page.get_text("text")
                    extracted_text.append(text)
            result_text = "\n".join(extracted_text)
            if not result_text.strip():
                print(f"Warning: No text extracted from {file_path}")
            return result_text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    async def run_comprehensive_analysis(self, cv_path: str, jd_path: str, 
                                       **kwargs) -> Dict[str, Any]:
        """Run comprehensive analysis with all enabled modules."""
        
        print(f"\n=== Starting Comprehensive CV-JD Analysis ===")
        print(f"Enabled modules: {', '.join(self.enabled_modules)}")
        
        # Extract text
        cv_text = await self.extract_pdf_text_tool(None, cv_path)
        jd_text = await self.extract_pdf_text_tool(None, jd_path)
        
        if not cv_text.strip() or not jd_text.strip():
            return {"error": "No text extracted from one or both PDF files."}
        
        # Run all enabled modules
        results = {}
        
        for module_name, module in self.modules.items():
            print(f"Running {module_name} analysis...")
            
            # Prepare module-specific kwargs
            module_kwargs = kwargs.copy()
            
            # Add analysis result for bias detection
            if module_name == "bias_detection" and "basic_matching" in results:
                module_kwargs["analysis_result"] = results["basic_matching"]
            
            # Run module analysis
            module_result = await module.analyze(cv_text, jd_text, **module_kwargs)
            results[module_name] = module_result
        
        # Store analysis history
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "cv_path": cv_path,
            "jd_path": jd_path,
            "enabled_modules": self.enabled_modules,
            "results": results
        })
        
        return results
    
    async def run(self, cv_path: str, jd_path: str, **kwargs):
        """Main function to run the enhanced CV and JD matching process."""
        if not (os.path.exists(cv_path) and os.path.exists(jd_path)):
            print("Error: One or both provided PDF paths do not exist.")
            return

        result = await self.run_comprehensive_analysis(cv_path, jd_path, **kwargs)

        if "error" in result:
            print(result["error"])
        else:
            self._print_comprehensive_results(result)
    
    def _print_comprehensive_results(self, results: Dict[str, Any]):
        """Print comprehensive analysis results."""
        
        for module_name, module_result in results.items():
            # Handle both string and dictionary responses for error checking
            if isinstance(module_result, str):
                if "error" in module_result.lower():
                    print(f"\n❌ {module_name.replace('_', ' ').title()}: {module_result}")
                    continue
                else:
                    print(f"\n✅ {module_name.replace('_', ' ').title()}:")
                    print(f"  Analysis: {module_result}")
                    continue
            elif isinstance(module_result, dict) and "error" in module_result:
                print(f"\n❌ {module_name.replace('_', ' ').title()}: {module_result['error']}")
                continue
            
            print(f"\n✅ {module_name.replace('_', ' ').title()}:")
            
            if module_name == "basic_matching":
                self._print_basic_matching_results(module_result)
            elif module_name == "culture_fit":
                self._print_culture_fit_results(module_result)
            elif module_name == "career_trajectory":
                self._print_career_trajectory_results(module_result)
            elif module_name == "market_intelligence":
                self._print_market_intelligence_results(module_result)
            elif module_name == "bias_detection":
                self._print_bias_detection_results(module_result)
            else:
                # Generic printing for other modules
                # Handle both string and dictionary responses
                if isinstance(module_result, str):
                    print(f"  Analysis: {module_result}")
                else:
                    for key, value in module_result.items():
                        if isinstance(value, list):
                            print(f"  {key.replace('_', ' ').title()}:")
                            for item in value:
                                print(f"    • {item}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n=== Enhanced Analysis Complete ===")
    
    def _print_basic_matching_results(self, result: Dict[str, Any]):
        """Print basic matching results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return
        
        print(f"  Match Score: {result.get('match_score', 'N/A')}%")
        print(f"  Match Category: {result.get('match_category', 'N/A')}")
        
        if result.get('matching_skills'):
            print("  Matching Skills:")
            for skill in result['matching_skills']:
                print(f"    • {skill}")
        
        if result.get('missing_requirements'):
            print("  Missing Requirements:")
            for req in result['missing_requirements']:
                print(f"    • {req}")
    
    def _print_culture_fit_results(self, result: Dict[str, Any]):
        """Print culture fit results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return
            
        print(f"  Culture Fit Score: {result.get('culture_fit_score', 'N/A')}%")
        print(f"  Communication Style: {result.get('communication_style_match', 'N/A')}")
        print(f"  Work Style Alignment: {result.get('work_style_alignment', 'N/A')}")
    
    def _print_career_trajectory_results(self, result: Dict[str, Any]):
        """Print career trajectory results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return
            
        print(f"  Career Growth Potential: {result.get('career_growth_potential', 'N/A')}%")
        print(f"  Predicted Next Roles: {result.get('predicted_next_roles', 'N/A')}")
        print(f"  Industry Relevance: {result.get('industry_relevance_score', 'N/A')}%")
    
    def _print_market_intelligence_results(self, result: Dict[str, Any]):
        """Print market intelligence results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return
            
        salary_range = result.get('salary_range', {})
        print(f"  Salary Range: ${salary_range.get('min', 'N/A')} - ${salary_range.get('max', 'N/A')}")
        
        market_trends = result.get('market_trends', {})
        print(f"  Market Demand: {'Growing' if market_trends.get('growing_demand') else 'Stable'}")
        print(f"  Remote Work Adoption: {market_trends.get('remote_work_adoption', 'N/A')}%")
    
    def _print_bias_detection_results(self, result: Dict[str, Any]):
        """Print bias detection results."""
        # Handle both string and dictionary responses
        if isinstance(result, str):
            print(f"  Analysis: {result}")
            return
            
        print(f"  Fairness Score: {result.get('fairness_score', 'N/A')}%")
        
        if result.get('bias_mitigation_recommendations'):
            print("  Bias Mitigation Recommendations:")
            for rec in result['bias_mitigation_recommendations']:
                print(f"    • {rec}")
    
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
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history
    
    def export_results(self, results: Dict[str, Any], format: str = "json") -> str:
        """Export results in specified format."""
        if format.lower() == "json":
            return json.dumps(results, indent=2)
        elif format.lower() == "csv":
            # Convert results to CSV format
            csv_data = []
            for module_name, module_result in results.items():
                if isinstance(module_result, dict):
                    for key, value in module_result.items():
                        csv_data.append([module_name, key, str(value)])
            
            df = pd.DataFrame(csv_data, columns=["Module", "Metric", "Value"])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")


async def main():
    """Main function to demonstrate the enhanced CV-JD matcher."""
    api_key = API_KEY
    
    # Configuration - easily customizable
    config = {
        "cv_path": "E:\\imageextractor\\imageextractor\\React_JS_5_Years.pdf",
        "jd_path": "E:\\imageextractor\\imageextractor\\React_Resume.pdf",
        "enabled_modules": [
            "basic_matching",
            "culture_fit",
            "career_trajectory",
            "resume_optimization",
            "interview_preparation",
            "bias_detection",
            "market_intelligence"
        ],
        "analysis_params": {
            "company_values": "Innovation, Collaboration, Continuous Learning, Customer Focus, Diversity",
            "industry": "Technology",
            "role": "Software Engineer",
            "location": "Remote"
        }
    }
    
    # Initialize matcher with configuration
    matcher = EnhancedCVJDMatcher(
        api_key=api_key,
        enabled_modules=config["enabled_modules"]
    )
    
    # Run analysis
    await matcher.run(
        cv_path=config["cv_path"],
        jd_path=config["jd_path"],
        **config["analysis_params"]
    )


if __name__ == "__main__":
    asyncio.run(main()) 