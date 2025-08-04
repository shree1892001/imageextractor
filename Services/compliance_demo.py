#!/usr/bin/env python3
"""
Intelligent Compliance Checker Demo
==================================

This demo showcases a unique and practical use case of pydantic-ai for:
- Legal document compliance analysis
- Financial compliance assessment
- Data privacy and GDPR compliance
- Regulatory compliance checking
- Contract risk assessment

The system provides comprehensive compliance analysis with:
- Automated issue detection
- Risk scoring and assessment
- Cost estimation for remediation
- Priority action recommendations
- Compliance timeline planning
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any, List

from IntelligentComplianceChecker import (
    IntelligentComplianceChecker,
    DocumentType,
    ComplianceLevel,
    RiskCategory
)

from imageextractor.Common.constants import API_KEY


class ComplianceDemo:
    """Demo class to showcase the Intelligent Compliance Checker."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.checker = IntelligentComplianceChecker(
            api_key=api_key,
            enabled_modules=[
                "legal_compliance",
                "financial_compliance", 
                "data_privacy_compliance",
                "regulatory_compliance",
                "contract_risk_assessment"
            ]
        )
    
    async def demo_contract_analysis(self):
        """Demo contract compliance analysis."""
        print("\n" + "="*60)
        print("📋 CONTRACT COMPLIANCE ANALYSIS DEMO")
        print("="*60)
        
        # Sample contract text (in real scenario, this would be extracted from PDF)
        sample_contract = """
        SOFTWARE DEVELOPMENT AGREEMENT
        
        This Agreement is entered into between TechCorp Inc. ("Client") and DevSolutions LLC ("Contractor") 
        effective January 1, 2024.
        
        ARTICLE 1: SERVICES
        Contractor shall provide software development services including but not limited to:
        - Web application development
        - Database design and implementation
        - API development and integration
        - Testing and quality assurance
        
        ARTICLE 2: COMPENSATION
        Client shall pay Contractor $150,000 for all services rendered.
        Payment terms: 50% upon contract signing, 50% upon project completion.
        
        ARTICLE 3: INTELLECTUAL PROPERTY
        All work product shall be owned by Client upon payment.
        Contractor retains rights to pre-existing intellectual property.
        
        ARTICLE 4: CONFIDENTIALITY
        Contractor agrees to maintain confidentiality of Client's proprietary information.
        This obligation survives termination of the agreement.
        
        ARTICLE 5: TERM AND TERMINATION
        This agreement shall commence on January 1, 2024 and continue until December 31, 2024.
        Either party may terminate with 30 days written notice.
        
        ARTICLE 6: DATA PROCESSING
        Contractor may process personal data as necessary to perform services.
        Contractor shall implement appropriate data protection measures.
        
        ARTICLE 7: GOVERNING LAW
        This agreement shall be governed by the laws of California.
        
        IN WITNESS WHEREOF, the parties have executed this agreement.
        """
        
        # Create temporary file for demo
        temp_file = "temp_contract.txt"
        with open(temp_file, "w") as f:
            f.write(sample_contract)
        
        try:
            # Run analysis
            analysis = await self.checker.analyze_document(
                document_path=temp_file,
                document_type=DocumentType.CONTRACT,
                jurisdiction="United States",
                industry="Technology",
                applicable_regulations=["GDPR", "CCPA", "SOX"],
                contract_value="$150,000",
                contract_duration="1 year"
            )
            
            # Print results
            self.checker.print_analysis_results(analysis)
            
            # Export results
            json_export = self.checker.export_analysis(analysis, "json")
            print(f"\n📄 Full JSON Export (first 1000 chars):")
            print(json_export[:1000] + "..." if len(json_export) > 1000 else json_export)
            
        except Exception as e:
            print(f"❌ Contract analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_privacy_policy_analysis(self):
        """Demo privacy policy compliance analysis."""
        print("\n" + "="*60)
        print("🔒 PRIVACY POLICY COMPLIANCE DEMO")
        print("="*60)
        
        # Sample privacy policy text
        sample_privacy_policy = """
        PRIVACY POLICY
        
        Last updated: January 1, 2024
        
        This Privacy Policy describes how TechCorp Inc. ("we", "us", "our") collects, uses, 
        and shares your personal information when you use our services.
        
        INFORMATION WE COLLECT
        We collect information you provide directly to us, such as:
        - Name and contact information
        - Payment information
        - Account credentials
        - Usage data and analytics
        
        HOW WE USE YOUR INFORMATION
        We use the information we collect to:
        - Provide and maintain our services
        - Process transactions
        - Send marketing communications
        - Improve our services
        
        SHARING YOUR INFORMATION
        We may share your information with:
        - Service providers and partners
        - Legal authorities when required
        - Third-party advertisers
        
        YOUR RIGHTS
        You have the right to:
        - Access your personal data
        - Correct inaccurate data
        - Request deletion of your data
        - Opt-out of marketing communications
        
        DATA RETENTION
        We retain your information for as long as necessary to provide our services.
        
        COOKIES AND TRACKING
        We use cookies and similar technologies to enhance your experience.
        
        CONTACT US
        For privacy-related questions, contact us at privacy@techcorp.com
        """
        
        # Create temporary file for demo
        temp_file = "temp_privacy_policy.txt"
        with open(temp_file, "w") as f:
            f.write(sample_privacy_policy)
        
        try:
            # Run analysis
            analysis = await self.checker.analyze_document(
                document_path=temp_file,
                document_type=DocumentType.PRIVACY_POLICY,
                jurisdiction="United States",
                industry="Technology",
                applicable_regulations=["GDPR", "CCPA", "PIPEDA"],
                data_types=["personal_data", "sensitive_data", "payment_data"]
            )
            
            # Print results
            self.checker.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Privacy policy analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_employment_agreement_analysis(self):
        """Demo employment agreement compliance analysis."""
        print("\n" + "="*60)
        print("👥 EMPLOYMENT AGREEMENT COMPLIANCE DEMO")
        print("="*60)
        
        # Sample employment agreement text
        sample_employment_agreement = """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into between TechCorp Inc. ("Company") and 
        John Doe ("Employee") effective March 1, 2024.
        
        POSITION AND DUTIES
        Employee shall serve as Senior Software Engineer with responsibilities including:
        - Software development and maintenance
        - Code review and quality assurance
        - Technical documentation
        - Team collaboration and mentoring
        
        COMPENSATION AND BENEFITS
        - Annual salary: $120,000
        - Health insurance coverage
        - 401(k) retirement plan
        - 20 days paid time off annually
        
        WORK SCHEDULE
        Employee shall work 40 hours per week, Monday through Friday.
        Remote work is permitted with manager approval.
        
        CONFIDENTIALITY AND INTELLECTUAL PROPERTY
        Employee agrees to maintain confidentiality of company information.
        All work product created during employment belongs to the Company.
        
        NON-COMPETE CLAUSE
        Employee agrees not to work for competitors for 12 months after termination.
        Geographic scope: California and neighboring states.
        
        TERMINATION
        Either party may terminate with 30 days written notice.
        Company may terminate immediately for cause.
        
        DISPUTE RESOLUTION
        Any disputes shall be resolved through binding arbitration in California.
        
        GOVERNING LAW
        This agreement is governed by California law.
        """
        
        # Create temporary file for demo
        temp_file = "temp_employment_agreement.txt"
        with open(temp_file, "w") as f:
            f.write(sample_employment_agreement)
        
        try:
            # Run analysis
            analysis = await self.checker.analyze_document(
                document_path=temp_file,
                document_type=DocumentType.EMPLOYMENT_AGREEMENT,
                jurisdiction="United States",
                industry="Technology",
                applicable_regulations=["FLSA", "ADA", "FMLA", "ERISA"],
                contract_value="$120,000",
                contract_duration="Indefinite"
            )
            
            # Print results
            self.checker.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Employment agreement analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_custom_compliance_module(self):
        """Demo custom compliance module creation."""
        print("\n" + "="*60)
        print("🔧 CUSTOM COMPLIANCE MODULE DEMO")
        print("="*60)
        
        from IntelligentComplianceChecker import ComplianceModule
        
        class CustomIndustryComplianceModule(ComplianceModule):
            """Custom module for industry-specific compliance."""
            
            def __init__(self, llm_model):
                self.llm = llm_model
                self.agent = Agent(
                    model=self.llm,
                    system_prompt="""
                    You are an expert in healthcare industry compliance specializing in:
                    - HIPAA compliance requirements
                    - FDA regulations for medical software
                    - Healthcare data security standards
                    - Medical device software regulations
                    - Patient privacy protection
                    """,
                    retries=2,
                )
            
            async def analyze(self, document_text: str, document_type: DocumentType, **kwargs) -> Dict[str, Any]:
                prompt = f"""
                Analyze the following {document_type.value} document for healthcare compliance:
                
                Document Text: {document_text}
                
                Provide healthcare compliance analysis in JSON format with:
                - hipaa_compliance_issues (list of HIPAA compliance issues)
                - fda_regulatory_requirements (FDA requirements if applicable)
                - data_security_gaps (security compliance gaps)
                - patient_privacy_concerns (privacy protection issues)
                - medical_device_considerations (if applicable)
                - compliance_recommendations (healthcare-specific recommendations)
                - risk_assessment (healthcare compliance risk score 0-100)
                """
                
                try:
                    response = await self.agent.run(prompt)
                    if hasattr(response, 'output'):
                        return response.output
                    elif hasattr(response, 'data'):
                        return response.data
                    elif hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                except Exception as e:
                    return {"error": f"Healthcare compliance analysis failed: {str(e)}"}
            
            def get_name(self) -> str:
                return "healthcare_compliance"
        
        # Add custom module to checker
        custom_module = CustomIndustryComplianceModule(self.checker.llm)
        self.checker.add_module(custom_module)
        
        print("✅ Custom healthcare compliance module added successfully!")
        print(f"Available modules: {', '.join(self.checker.enabled_modules)}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive compliance analysis demo."""
        print("🚀 INTELLIGENT COMPLIANCE CHECKER DEMO")
        print("="*60)
        print("This demo showcases a unique and practical use case of pydantic-ai")
        print("for automated document compliance analysis and risk assessment.")
        print("="*60)
        
        # Run all demos
        await self.demo_contract_analysis()
        await self.demo_privacy_policy_analysis()
        await self.demo_employment_agreement_analysis()
        await self.demo_custom_compliance_module()
        
        # Show analysis history
        print("\n" + "="*60)
        print("📊 ANALYSIS HISTORY")
        print("="*60)
        
        history = self.checker.get_analysis_history()
        for i, entry in enumerate(history, 1):
            print(f"\n{i}. Analysis Entry:")
            print(f"   Timestamp: {entry['timestamp']}")
            print(f"   Document Type: {entry['document_type']}")
            print(f"   Modules Used: {', '.join(entry['enabled_modules'])}")
        
        print(f"\n✅ Demo completed successfully! Total analyses: {len(history)}")


async def main():
    """Main function to run the compliance demo."""
    api_key = API_KEY
    
    if not api_key:
        print("❌ API key not found. Please set your API key in constants.py")
        return
    
    # Initialize and run demo
    demo = ComplianceDemo(api_key)
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main()) 