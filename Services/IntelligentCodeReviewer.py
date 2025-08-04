import asyncio
import os
import json
import ast
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import base64

from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from imageextractor.Common.constants import API_KEY


class SecurityLevel(str, Enum):
    """Security levels for vulnerability assessment."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    SAFE = "safe"


class CodeLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class IssueCategory(str, Enum):
    """Categories of code issues."""
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_QUALITY = "code_quality"
    BEST_PRACTICE = "best_practice"
    MAINTAINABILITY = "maintainability"
    ACCESSIBILITY = "accessibility"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class SecurityVulnerability(BaseModel):
    """Model for security vulnerabilities found in code."""
    vulnerability_id: str = Field(..., description="Unique identifier for the vulnerability")
    category: IssueCategory = Field(..., description="Category of the issue")
    severity: SecurityLevel = Field(..., description="Severity level of the vulnerability")
    title: str = Field(..., description="Brief title of the vulnerability")
    description: str = Field(..., description="Detailed description of the vulnerability")
    location: str = Field(..., description="Where in the code the vulnerability was found")
    code_snippet: str = Field(..., description="Relevant code snippet")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    cvss_score: Optional[float] = Field(None, ge=0, le=10, description="CVSS score if applicable")
    remediation: str = Field(..., description="Recommended fix for the vulnerability")
    impact: str = Field(..., description="Potential impact of the vulnerability")
    references: List[str] = Field(default_factory=list, description="Reference links for more information")
    
    @field_validator('severity')
    def validate_severity(cls, v):
        if v not in SecurityLevel:
            raise ValueError(f"Invalid severity level: {v}")
        return v


class CodeAnalysis(BaseModel):
    """Model for comprehensive code analysis."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    file_path: str = Field(..., description="Path to the analyzed file")
    language: CodeLanguage = Field(..., description="Programming language of the code")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    overall_security_score: float = Field(..., ge=0, le=100, description="Overall security score (0-100)")
    overall_quality_score: float = Field(..., ge=0, le=100, description="Overall code quality score (0-100)")
    overall_performance_score: float = Field(..., ge=0, le=100, description="Overall performance score (0-100)")
    risk_level: SecurityLevel = Field(..., description="Overall risk level")
    security_vulnerabilities: List[SecurityVulnerability] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    estimated_fix_time: str = Field(..., description="Estimated time to fix all issues")
    priority_actions: List[str] = Field(default_factory=list)
    code_metrics: Dict[str, Any] = Field(default_factory=dict, description="Code complexity and metrics")
    
    @field_validator('overall_security_score', 'overall_quality_score', 'overall_performance_score')
    def validate_scores(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Scores must be between 0 and 100")
        return v


class AnalysisModule(ABC):
    """Abstract base class for code analysis modules."""
    
    @abstractmethod
    async def analyze(self, code_content: str, language: CodeLanguage, **kwargs) -> Dict[str, Any]:
        """Perform code analysis and return results."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this analysis module."""
        pass 