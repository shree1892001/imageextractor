import asyncio
import os
import json
import fitz
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


class ComplianceLevel(str, Enum):
    """Compliance levels for risk assessment."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    COMPLIANT = "compliant"


class DocumentType(str, Enum):
    """Types of documents that can be analyzed."""
    CONTRACT = "contract"
    LEGAL_AGREEMENT = "legal_agreement"
    PRIVACY_POLICY = "privacy_policy"
    TERMS_OF_SERVICE = "terms_of_service"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    NDA = "nda"
    LICENSE_AGREEMENT = "license_agreement"
    REGULATORY_FILING = "regulatory_filing"
    COMPLIANCE_REPORT = "compliance_report"
    FINANCIAL_DOCUMENT = "financial_document"


class RiskCategory(str, Enum):
    """Categories of compliance risks."""
    LEGAL_RISK = "legal_risk"
    FINANCIAL_RISK = "financial_risk"
    OPERATIONAL_RISK = "operational_risk"
    REPUTATIONAL_RISK = "reputational_risk"
    REGULATORY_RISK = "regulatory_risk"
    DATA_PRIVACY_RISK = "data_privacy_risk"
    CONTRACTUAL_RISK = "contractual_risk"


class ComplianceIssue(BaseModel):
    """Model for compliance issues found in documents."""
    issue_id: str = Field(..., description="Unique identifier for the issue")
    category: RiskCategory = Field(..., description="Category of the risk")
    severity: ComplianceLevel = Field(..., description="Severity level of the issue")
    title: str = Field(..., description="Brief title of the issue")
    description: str = Field(..., description="Detailed description of the issue")
    location: str = Field(..., description="Where in the document the issue was found")
    relevant_text: str = Field(..., description="Relevant text from the document")
    recommendation: str = Field(..., description="Recommended action to address the issue")
    legal_basis: str = Field(..., description="Legal basis for the compliance requirement")
    deadline: Optional[str] = Field(None, description="Deadline for addressing the issue")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost to address the issue")
    
    @field_validator('severity')
    def validate_severity(cls, v):
        if v not in ComplianceLevel:
            raise ValueError(f"Invalid severity level: {v}")
        return v


class DocumentAnalysis(BaseModel):
    """Model for comprehensive document analysis."""
    document_id: str = Field(..., description="Unique identifier for the document")
    document_type: DocumentType = Field(..., description="Type of document analyzed")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    overall_compliance_score: float = Field(..., ge=0, le=100, description="Overall compliance score (0-100)")
    risk_level: ComplianceLevel = Field(..., description="Overall risk level")
    compliance_issues: List[ComplianceIssue] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    estimated_remediation_cost: float = Field(0, description="Estimated cost to achieve full compliance")
    priority_actions: List[str] = Field(default_factory=list)
    compliance_timeline: str = Field(..., description="Recommended timeline for achieving compliance")
    
    @field_validator('overall_compliance_score')
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Compliance score must be between 0 and 100")
        return v


class ComplianceModule(ABC):
    """Abstract base class for compliance analysis modules."""
    
    @abstractmethod
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        """Perform compliance analysis and return results."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this compliance module."""
        pass


class LegalComplianceModule(ComplianceModule):
    """Legal compliance analysis module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert legal compliance analyst specializing in contract law, 
            regulatory compliance, and risk assessment. Your role is to:
            - Identify potential legal issues and compliance gaps
            - Assess risks based on applicable laws and regulations
            - Provide specific recommendations for compliance
            - Evaluate contractual obligations and liabilities
            - Identify missing legal protections and clauses
            """,
            retries=2,
        )
    
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        jurisdiction = kwargs.get('jurisdiction', 'United States')
        industry = kwargs.get('industry', 'Technology')
        
        prompt = f"""
        Analyze the following {document_type.value} document for legal compliance issues:
        
        Document Text: {document_text}
        Jurisdiction: {jurisdiction}
        Industry: {industry}
        
        Provide a comprehensive legal analysis in JSON format with:
        - legal_issues (list of legal compliance issues)
        - missing_clauses (list of missing legal clauses)
        - risk_assessment (overall risk level and specific risks)
        - regulatory_requirements (applicable regulations)
        - recommendations (specific legal recommendations)
        - estimated_legal_risk (risk score 0-100)
        - compliance_deadlines (any applicable deadlines)
        - potential_penalties (potential legal consequences)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Legal compliance analysis failed: {str(e)}"}
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        """Extract data from agent response."""
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'data'):
            return response.data
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def get_name(self) -> str:
        return "legal_compliance"


class FinancialComplianceModule(ComplianceModule):
    """Financial compliance analysis module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert financial compliance analyst specializing in:
            - Financial reporting requirements
            - Tax compliance and obligations
            - Anti-money laundering (AML) regulations
            - Know Your Customer (KYC) requirements
            - Financial risk assessment
            - Audit trail requirements
            """,
            retries=2,
        )
    
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        financial_year = kwargs.get('financial_year', '2024')
        reporting_standard = kwargs.get('reporting_standard', 'GAAP')
        
        prompt = f"""
        Analyze the following {document_type.value} document for financial compliance:
        
        Document Text: {document_text}
        Financial Year: {financial_year}
        Reporting Standard: {reporting_standard}
        
        Provide financial compliance analysis in JSON format with:
        - financial_obligations (list of financial obligations)
        - tax_implications (tax-related compliance issues)
        - reporting_requirements (financial reporting obligations)
        - audit_requirements (audit and documentation needs)
        - financial_risks (financial risk assessment)
        - compliance_gaps (missing financial compliance elements)
        - estimated_financial_impact (cost of non-compliance)
        - reporting_deadlines (financial reporting deadlines)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Financial compliance analysis failed: {str(e)}"}
    
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
        return "financial_compliance"


class DataPrivacyComplianceModule(ComplianceModule):
    """Data privacy and GDPR compliance analysis module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert data privacy and GDPR compliance analyst specializing in:
            - GDPR compliance requirements
            - Data protection regulations
            - Privacy policy analysis
            - Data processing agreements
            - Consent management
            - Data subject rights
            - Cross-border data transfers
            """,
            retries=2,
        )
    
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        applicable_regulations = kwargs.get('applicable_regulations', ['GDPR', 'CCPA', 'PIPEDA'])
        data_types = kwargs.get('data_types', ['personal_data', 'sensitive_data'])
        
        prompt = f"""
        Analyze the following {document_type.value} document for data privacy compliance:
        
        Document Text: {document_text}
        Applicable Regulations: {', '.join(applicable_regulations)}
        Data Types: {', '.join(data_types)}
        
        Provide data privacy compliance analysis in JSON format with:
        - privacy_issues (list of privacy compliance issues)
        - data_processing_activities (identified data processing)
        - consent_requirements (consent management issues)
        - data_subject_rights (rights implementation)
        - cross_border_transfers (international data transfers)
        - data_retention_policies (retention and deletion)
        - breach_notification_requirements (incident response)
        - privacy_impact_assessment (PIA requirements)
        - compliance_gaps (missing privacy protections)
        - recommended_actions (privacy compliance actions)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Data privacy compliance analysis failed: {str(e)}"}
    
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
        return "data_privacy_compliance"


class RegulatoryComplianceModule(ComplianceModule):
    """Industry-specific regulatory compliance analysis module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert regulatory compliance analyst specializing in:
            - Industry-specific regulations (finance, healthcare, technology)
            - Government compliance requirements
            - Industry standards and best practices
            - Regulatory reporting obligations
            - Compliance monitoring and auditing
            - Regulatory risk assessment
            """,
            retries=2,
        )
    
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        industry = kwargs.get('industry', 'Technology')
        jurisdiction = kwargs.get('jurisdiction', 'United States')
        regulatory_frameworks = kwargs.get('regulatory_frameworks', ['SOX', 'HIPAA', 'PCI-DSS'])
        
        prompt = f"""
        Analyze the following {document_type.value} document for regulatory compliance:
        
        Document Text: {document_text}
        Industry: {industry}
        Jurisdiction: {jurisdiction}
        Regulatory Frameworks: {', '.join(regulatory_frameworks)}
        
        Provide regulatory compliance analysis in JSON format with:
        - regulatory_requirements (applicable regulations)
        - compliance_gaps (missing regulatory compliance)
        - reporting_obligations (regulatory reporting needs)
        - audit_requirements (compliance audit needs)
        - enforcement_risks (regulatory enforcement risks)
        - industry_standards (industry-specific requirements)
        - compliance_timeline (regulatory compliance timeline)
        - penalty_assessment (potential regulatory penalties)
        - recommended_actions (regulatory compliance actions)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Regulatory compliance analysis failed: {str(e)}"}
    
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
        return "regulatory_compliance"


class ContractRiskAssessmentModule(ComplianceModule):
    """Contract-specific risk assessment module."""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.agent = Agent(
            model=self.llm,
            system_prompt="""
            You are an expert contract risk analyst specializing in:
            - Contractual risk assessment
            - Liability analysis
            - Performance risk evaluation
            - Financial risk in contracts
            - Termination risk assessment
            - Dispute resolution analysis
            - Contract optimization recommendations
            """,
            retries=2,
        )
    
    async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
        contract_value = kwargs.get('contract_value', 'Unknown')
        contract_duration = kwargs.get('contract_duration', 'Unknown')
        
        prompt = f"""
        Analyze the following {document_type.value} document for contractual risks:
        
        Document Text: {document_text}
        Contract Value: {contract_value}
        Contract Duration: {contract_duration}
        
        Provide contract risk assessment in JSON format with:
        - contractual_risks (list of contractual risk factors)
        - liability_assessment (liability exposure analysis)
        - performance_risks (performance-related risks)
        - financial_exposure (financial risk assessment)
        - termination_risks (termination-related risks)
        - dispute_risks (dispute resolution risks)
        - risk_mitigation_strategies (risk reduction strategies)
        - contract_optimization_recommendations (improvement suggestions)
        - risk_score (overall risk score 0-100)
        """
        
        try:
            response = await self.agent.run(prompt)
            return self._extract_response_data(response)
        except Exception as e:
            return {"error": f"Contract risk assessment failed: {str(e)}"}
    
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
        return "contract_risk_assessment"


class IntelligentComplianceChecker:
    """Intelligent Document Processing and Compliance Checker."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", 
                 enabled_modules: List[str] = None):
        """
        Initialize the Intelligent Compliance Checker.
        
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
            "legal_compliance",
            "financial_compliance",
            "data_privacy_compliance",
            "regulatory_compliance",
            "contract_risk_assessment"
        ]
    
    def _initialize_modules(self) -> Dict[str, ComplianceModule]:
        """Initialize compliance modules based on configuration."""
        module_classes = {
            "legal_compliance": LegalComplianceModule,
            "financial_compliance": FinancialComplianceModule,
            "data_privacy_compliance": DataPrivacyComplianceModule,
            "regulatory_compliance": RegulatoryComplianceModule,
            "contract_risk_assessment": ContractRiskAssessmentModule
        }
        
        modules = {}
        for module_name in self.enabled_modules:
            if module_name in module_classes:
                modules[module_name] = module_classes[module_name](self.llm)
        
        return modules
    
    async def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF document."""
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
    
    async def analyze_document(self, document_path: str, document_type: DocumentType, 
                              **kwargs) -> DocumentAnalysis:
        """Analyze a document for compliance issues."""
        
        print(f"\n=== Starting Intelligent Compliance Analysis ===")
        print(f"Document Type: {document_type.value}")
        print(f"Enabled modules: {', '.join(self.enabled_modules)}")
        
        # Extract text from document
        document_text = await self.extract_pdf_text(document_path)
        
        if not document_text.strip():
            raise ValueError("No text extracted from the document.")
        
        # Run all enabled modules
        module_results = {}
        
        for module_name, module in self.modules.items():
            print(f"Running {module_name} analysis...")
            
            module_kwargs = kwargs.copy()
            module_result = await module.analyze(document_text, document_type, **module_kwargs)
            module_results[module_name] = module_result
        
        # Compile comprehensive analysis
        analysis = await self._compile_analysis(document_path, document_type, module_results, **kwargs)
        
        # Store analysis history
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "document_path": document_path,
            "document_type": document_type.value,
            "enabled_modules": self.enabled_modules,
            "results": analysis.model_dump()
        })
        
        return analysis
    
    async def _compile_analysis(self, document_path: str, document_type: DocumentType, 
                               module_results: Dict[str, Any], **kwargs) -> DocumentAnalysis:
        """Compile results from all modules into a comprehensive analysis."""
        
        # Generate unique document ID
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Compile all compliance issues
        all_issues = []
        all_recommendations = []
        total_remediation_cost = 0.0
        
        for module_name, result in module_results.items():
            if isinstance(result, dict) and "error" not in result:
                # Extract issues from module results
                if "legal_issues" in result:
                    for issue in result["legal_issues"]:
                        all_issues.append(ComplianceIssue(
                            issue_id=f"{module_name}_{len(all_issues)}",
                            category=RiskCategory.LEGAL_RISK,
                            severity=ComplianceLevel.HIGH,
                            title=issue.get("title", "Legal Issue"),
                            description=issue.get("description", ""),
                            location=issue.get("location", "Document"),
                            relevant_text=issue.get("relevant_text", ""),
                            recommendation=issue.get("recommendation", ""),
                            legal_basis=issue.get("legal_basis", ""),
                            estimated_cost=issue.get("estimated_cost", 0.0)
                        ))
                
                # Extract recommendations
                if "recommendations" in result:
                    all_recommendations.extend(result["recommendations"])
                
                # Calculate remediation costs
                if "estimated_financial_impact" in result:
                    total_remediation_cost += result["estimated_financial_impact"]
        
        # Calculate overall compliance score
        total_issues = len(all_issues)
        critical_issues = len([i for i in all_issues if i.severity == ComplianceLevel.CRITICAL])
        high_issues = len([i for i in all_issues if i.severity == ComplianceLevel.HIGH])
        
        # Simple scoring algorithm
        base_score = 100
        score_deduction = (critical_issues * 20) + (high_issues * 10) + (total_issues * 5)
        compliance_score = max(0, base_score - score_deduction)
        
        # Determine overall risk level
        if critical_issues > 0:
            risk_level = ComplianceLevel.CRITICAL
        elif high_issues > 2:
            risk_level = ComplianceLevel.HIGH
        elif total_issues > 5:
            risk_level = ComplianceLevel.MEDIUM
        elif total_issues > 0:
            risk_level = ComplianceLevel.LOW
        else:
            risk_level = ComplianceLevel.COMPLIANT
        
        # Generate priority actions
        priority_actions = []
        if critical_issues > 0:
            priority_actions.append("Address critical compliance issues immediately")
        if high_issues > 0:
            priority_actions.append("Review and resolve high-priority compliance gaps")
        if total_remediation_cost > 0:
            priority_actions.append(f"Allocate budget for compliance remediation (${total_remediation_cost:,.2f})")
        
        # Determine compliance timeline
        if critical_issues > 0:
            timeline = "Immediate action required (30 days)"
        elif high_issues > 0:
            timeline = "High priority (60 days)"
        elif total_issues > 0:
            timeline = "Standard priority (90 days)"
        else:
            timeline = "Compliant - maintain current standards"
        
        return DocumentAnalysis(
            document_id=document_id,
            document_type=document_type,
            overall_compliance_score=compliance_score,
            risk_level=risk_level,
            compliance_issues=all_issues,
            recommendations=all_recommendations,
            estimated_remediation_cost=total_remediation_cost,
            priority_actions=priority_actions,
            compliance_timeline=timeline
        )
    
    def print_analysis_results(self, analysis: DocumentAnalysis):
        """Print comprehensive analysis results."""
        
        print(f"\n=== Compliance Analysis Results ===")
        print(f"Document ID: {analysis.document_id}")
        print(f"Document Type: {analysis.document_type.value}")
        print(f"Analysis Date: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Compliance Score: {analysis.overall_compliance_score:.1f}%")
        print(f"Risk Level: {analysis.risk_level.value.upper()}")
        print(f"Estimated Remediation Cost: ${analysis.estimated_remediation_cost:,.2f}")
        print(f"Compliance Timeline: {analysis.compliance_timeline}")
        
        if analysis.compliance_issues:
            print(f"\n📋 Compliance Issues Found ({len(analysis.compliance_issues)}):")
            for i, issue in enumerate(analysis.compliance_issues, 1):
                print(f"  {i}. [{issue.severity.value.upper()}] {issue.title}")
                print(f"     Category: {issue.category.value}")
                print(f"     Description: {issue.description}")
                print(f"     Recommendation: {issue.recommendation}")
                if issue.estimated_cost:
                    print(f"     Estimated Cost: ${issue.estimated_cost:,.2f}")
                print()
        
        if analysis.recommendations:
            print(f"\n💡 Recommendations ({len(analysis.recommendations)}):")
            for i, rec in enumerate(analysis.recommendations, 1):
                print(f"  {i}. {rec}")
        
        if analysis.priority_actions:
            print(f"\n🚨 Priority Actions:")
            for action in analysis.priority_actions:
                print(f"  • {action}")
        
        print(f"\n=== Analysis Complete ===")
    
    def export_analysis(self, analysis: DocumentAnalysis, format: str = "json") -> str:
        """Export analysis results in specified format."""
        if format.lower() == "json":
            return analysis.model_dump_json(indent=2)
        elif format.lower() == "csv":
            # Convert analysis to CSV format
            csv_data = []
            
            # Add main analysis data
            csv_data.append(["Document ID", analysis.document_id])
            csv_data.append(["Document Type", analysis.document_type.value])
            csv_data.append(["Compliance Score", f"{analysis.overall_compliance_score:.1f}%"])
            csv_data.append(["Risk Level", analysis.risk_level.value])
            csv_data.append(["Remediation Cost", f"${analysis.estimated_remediation_cost:,.2f}"])
            csv_data.append(["Timeline", analysis.compliance_timeline])
            
            # Add issues
            for issue in analysis.compliance_issues:
                csv_data.append([
                    "Issue",
                    issue.severity.value,
                    issue.category.value,
                    issue.title,
                    issue.description,
                    issue.recommendation,
                    f"${issue.estimated_cost:,.2f}" if issue.estimated_cost else "N/A"
                ])
            
            df = pd.DataFrame(csv_data, columns=["Type", "Severity", "Category", "Title", "Description", "Recommendation", "Cost"])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history
    
    def add_module(self, module: ComplianceModule):
        """Add a custom compliance module."""
        self.modules[module.get_name()] = module
        self.enabled_modules.append(module.get_name())
    
    def remove_module(self, module_name: str):
        """Remove a compliance module."""
        if module_name in self.modules:
            del self.modules[module_name]
            if module_name in self.enabled_modules:
                self.enabled_modules.remove(module_name)


async def main():
    """Main function to demonstrate the Intelligent Compliance Checker."""
    api_key = API_KEY
    
    # Configuration
    config = {
        "document_path": "E:\\imageextractor\\imageextractor\\hrjd.pdf",  # Replace with actual document path
        "document_type": DocumentType.CONTRACT,
        "enabled_modules": [
            "legal_compliance",
            "financial_compliance",
            "data_privacy_compliance",
            "regulatory_compliance",
            "contract_risk_assessment"
        ],
        "analysis_params": {
            "jurisdiction": "United States",
            "industry": "Technology",
            "applicable_regulations": ["GDPR", "CCPA", "SOX"],
            "contract_value": "$500,000",
            "contract_duration": "2 years"
        }
    }
    
    # Initialize compliance checker
    checker = IntelligentComplianceChecker(
        api_key=api_key,
        enabled_modules=config["enabled_modules"]
    )
    
    try:
        # Run analysis
        analysis = await checker.analyze_document(
            document_path=config["document_path"],
            document_type=config["document_type"],
            **config["analysis_params"]
        )
        
        # Print results
        checker.print_analysis_results(analysis)
        
        # Export results
        json_export = checker.export_analysis(analysis, "json")
        print(f"\nJSON Export (first 500 chars):\n{json_export[:500]}...")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 