"""
Configuration file for the Enhanced JD vs CV Matching System.
This file contains all configurable parameters and settings.
"""

import os
from typing import Dict, Any, List

# API Configuration
API_CONFIG = {
    "model": "gemini-1.5-flash",
    "retries": 2,
    "timeout": 30
}

# Default Analysis Modules
DEFAULT_MODULES = [
    "basic_matching",
    "culture_fit", 
    "career_trajectory",
    "resume_optimization",
    "interview_preparation",
    "bias_detection",
    "market_intelligence"
]

# Module-specific configurations
MODULE_CONFIGS = {
    "basic_matching": {
        "enabled": True,
        "priority": 1,
        "system_prompt": """
        You are an expert recruiter AI analyzing resumes and job descriptions.
        Your task is to:
        - Extract and match the relevant skills, qualifications, and experiences from the CV and JD.
        - Exclude any irrelevant information such as POS tags, common words, and other non-skill-related content.
        - Identify missing skills or gaps in the CV based on JD requirements.
        - Provide a match score (0-100%) and a comprehensive assessment.
        """
    },
    "culture_fit": {
        "enabled": True,
        "priority": 2,
        "system_prompt": """
        You are an expert in organizational psychology and culture analysis.
        Analyze communication styles, values, and cultural fit between candidates and organizations.
        Focus on soft skills, work style preferences, and cultural alignment.
        """
    },
    "career_trajectory": {
        "enabled": True,
        "priority": 3,
        "system_prompt": """
        You are an expert career counselor and industry analyst.
        Analyze career progression patterns, industry trends, and predict future skill requirements.
        Provide insights on career trajectory and growth potential.
        """
    },
    "resume_optimization": {
        "enabled": True,
        "priority": 4,
        "system_prompt": """
        You are an expert resume writer and career coach.
        Optimize resumes for specific job descriptions by improving keywords, structure, and content.
        """
    },
    "interview_preparation": {
        "enabled": True,
        "priority": 5,
        "system_prompt": """
        You are an expert interview coach and HR professional.
        Generate personalized interview questions, technical assessments, and behavioral scenarios.
        Focus on role-specific and candidate-specific questions.
        """
    },
    "bias_detection": {
        "enabled": True,
        "priority": 6,
        "system_prompt": """
        You are an expert in fair hiring practices and bias detection.
        Analyze recruitment processes for potential biases and provide mitigation strategies.
        """
    },
    "market_intelligence": {
        "enabled": True,
        "priority": 7,
        "data_sources": {
            "salary_data": "https://api.salary.com/v1/salary",
            "job_market": "https://api.indeed.com/v1/jobs",
            "skill_demand": "https://api.linkedin.com/v2/skills"
        }
    },
    "visual_analysis": {
        "enabled": False,
        "priority": 8,
        "image_formats": [".jpg", ".jpeg", ".png", ".pdf"],
        "max_image_size": 10 * 1024 * 1024  # 10MB
    }
}

# Industry-specific configurations
INDUSTRY_CONFIGS = {
    "technology": {
        "default_skills": ["python", "javascript", "react", "aws", "docker"],
        "salary_multiplier": 1.2,
        "growth_rate": "high",
        "remote_friendly": True
    },
    "finance": {
        "default_skills": ["excel", "sql", "python", "risk_management", "financial_modeling"],
        "salary_multiplier": 1.1,
        "growth_rate": "medium",
        "remote_friendly": False
    },
    "healthcare": {
        "default_skills": ["patient_care", "medical_terminology", "healthcare_systems"],
        "salary_multiplier": 1.0,
        "growth_rate": "high",
        "remote_friendly": False
    },
    "marketing": {
        "default_skills": ["digital_marketing", "seo", "social_media", "analytics"],
        "salary_multiplier": 0.9,
        "growth_rate": "medium",
        "remote_friendly": True
    }
}

# Company culture templates
CULTURE_TEMPLATES = {
    "startup": {
        "values": ["Innovation", "Fast-paced", "Risk-taking", "Flexibility"],
        "work_style": "Collaborative and dynamic",
        "growth_opportunities": "High"
    },
    "enterprise": {
        "values": ["Stability", "Process-driven", "Professional development", "Work-life balance"],
        "work_style": "Structured and methodical",
        "growth_opportunities": "Medium"
    },
    "consulting": {
        "values": ["Client focus", "Problem-solving", "Travel", "High performance"],
        "work_style": "Results-oriented and client-focused",
        "growth_opportunities": "High"
    }
}

# Export formats
EXPORT_FORMATS = {
    "json": {
        "enabled": True,
        "indent": 2,
        "encoding": "utf-8"
    },
    "csv": {
        "enabled": True,
        "delimiter": ",",
        "encoding": "utf-8"
    },
    "pdf": {
        "enabled": False,
        "template": "default_report.html"
    },
    "excel": {
        "enabled": False,
        "sheet_name": "Analysis Results"
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "charts": {
        "skill_radar": True,
        "matching_heatmap": True,
        "career_timeline": True,
        "salary_comparison": True
    },
    "colors": {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336"
    },
    "chart_style": "plotly"
}

# File paths and directories
PATHS = {
    "output_dir": "results",
    "logs_dir": "logs",
    "temp_dir": "temp",
    "templates_dir": "templates",
    "cache_dir": "cache"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "cv_jd_analysis.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Performance settings
PERFORMANCE_CONFIG = {
    "max_concurrent_analyses": 5,
    "timeout_per_module": 60,
    "cache_results": True,
    "cache_ttl": 3600,  # 1 hour
    "retry_attempts": 3
}

# Custom analysis parameters
CUSTOM_PARAMS = {
    "match_score_threshold": 70,
    "culture_fit_threshold": 60,
    "career_growth_threshold": 50,
    "bias_detection_sensitivity": "medium",
    "market_data_freshness_days": 30
}

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "cache_enabled": False,
        "mock_external_apis": True
    },
    "production": {
        "debug": False,
        "log_level": "WARNING",
        "cache_enabled": True,
        "mock_external_apis": False
    },
    "testing": {
        "debug": True,
        "log_level": "DEBUG",
        "cache_enabled": False,
        "mock_external_apis": True
    }
}

def get_config(environment: str = "development") -> Dict[str, Any]:
    """
    Get complete configuration for the specified environment.
    
    Args:
        environment: Environment name (development, production, testing)
    
    Returns:
        Complete configuration dictionary
    """
    base_config = {
        "api": API_CONFIG,
        "modules": DEFAULT_MODULES,
        "module_configs": MODULE_CONFIGS,
        "industry_configs": INDUSTRY_CONFIGS,
        "culture_templates": CULTURE_TEMPLATES,
        "export_formats": EXPORT_FORMATS,
        "visualization": VISUALIZATION_CONFIG,
        "paths": PATHS,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "custom_params": CUSTOM_PARAMS
    }
    
    # Merge with environment-specific config
    env_config = ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS["development"])
    base_config["environment"] = env_config
    
    return base_config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields and values.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["api", "modules", "module_configs"]
    
    for field in required_fields:
        if field not in config:
            print(f"Missing required configuration field: {field}")
            return False
    
    # Validate API configuration
    api_config = config.get("api", {})
    if "model" not in api_config:
        print("Missing API model configuration")
        return False
    
    # Validate module configurations
    module_configs = config.get("module_configs", {})
    for module_name, module_config in module_configs.items():
        if not isinstance(module_config, dict):
            print(f"Invalid module configuration for {module_name}")
            return False
    
    return True

def create_custom_config(**kwargs) -> Dict[str, Any]:
    """
    Create a custom configuration with overrides.
    
    Args:
        **kwargs: Configuration overrides
    
    Returns:
        Custom configuration dictionary
    """
    base_config = get_config()
    
    # Apply overrides
    for key, value in kwargs.items():
        if key in base_config:
            if isinstance(value, dict) and isinstance(base_config[key], dict):
                base_config[key].update(value)
            else:
                base_config[key] = value
        else:
            base_config[key] = value
    
    return base_config 