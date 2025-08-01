"""
Example usage of the Enhanced JD vs CV Matching System.
This file demonstrates various ways to use the system with different configurations.
"""

import asyncio
import json
from typing import Dict, Any
from JDVsCV_Enhanced import EnhancedCVJDMatcher
from config import get_config, create_custom_config, validate_config


async def basic_usage_example():
    """Basic usage example with default configuration."""
    print("=== Basic Usage Example ===")
    
    # Get default configuration
    config = get_config("development")
    
    # Initialize matcher
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=config["modules"]
    )
    
    # Run analysis
    await matcher.run(
        cv_path="path/to/candidate_resume.pdf",
        jd_path="path/to/job_description.pdf",
        company_values="Innovation, Collaboration, Continuous Learning",
        industry="Technology",
        role="Software Engineer",
        location="Remote"
    )


async def custom_modules_example():
    """Example with custom module selection."""
    print("=== Custom Modules Example ===")
    
    # Select only specific modules
    custom_modules = [
        "basic_matching",
        "culture_fit",
        "market_intelligence"
    ]
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=custom_modules
    )
    
    await matcher.run(
        cv_path="path/to/candidate_resume.pdf",
        jd_path="path/to/job_description.pdf",
        company_values="Customer Focus, Quality, Teamwork"
    )


async def industry_specific_example():
    """Example with industry-specific configuration."""
    print("=== Industry-Specific Example ===")
    
    # Get configuration for finance industry
    config = create_custom_config(
        industry="finance",
        custom_params={
            "match_score_threshold": 75,
            "culture_fit_threshold": 70
        }
    )
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=config["modules"]
    )
    
    await matcher.run(
        cv_path="path/to/finance_candidate.pdf",
        jd_path="path/to/finance_job.pdf",
        industry="Finance",
        role="Financial Analyst",
        location="New York"
    )


async def batch_analysis_example():
    """Example of batch analysis for multiple candidates."""
    print("=== Batch Analysis Example ===")
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=["basic_matching", "market_intelligence"]
    )
    
    # Multiple candidates and jobs
    candidates = [
        "path/to/candidate1.pdf",
        "path/to/candidate2.pdf",
        "path/to/candidate3.pdf"
    ]
    
    jobs = [
        "path/to/job1.pdf",
        "path/to/job2.pdf"
    ]
    
    results = {}
    
    for cv_path in candidates:
        for jd_path in jobs:
            result = await matcher.run_comprehensive_analysis(
                cv_path=cv_path,
                jd_path=jd_path,
                role="Software Engineer",
                location="Remote"
            )
            results[f"{cv_path}_{jd_path}"] = result
    
    # Export results
    json_results = matcher.export_results(results, "json")
    with open("batch_analysis_results.json", "w") as f:
        f.write(json_results)


async def custom_analysis_example():
    """Example with custom analysis parameters."""
    print("=== Custom Analysis Example ===")
    
    # Create custom configuration
    custom_config = create_custom_config(
        custom_params={
            "match_score_threshold": 80,
            "culture_fit_threshold": 75,
            "career_growth_threshold": 60,
            "bias_detection_sensitivity": "high"
        },
        performance={
            "timeout_per_module": 120,
            "retry_attempts": 5
        }
    )
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=custom_config["modules"]
    )
    
    # Run with custom parameters
    result = await matcher.run_comprehensive_analysis(
        cv_path="path/to/senior_candidate.pdf",
        jd_path="path/to/senior_position.pdf",
        company_values="Leadership, Innovation, Excellence",
        industry="Technology",
        role="Senior Software Engineer",
        location="San Francisco"
    )
    
    # Export detailed results
    detailed_report = matcher.export_results(result, "json")
    with open("detailed_analysis_report.json", "w") as f:
        f.write(detailed_report)


async def startup_recruitment_example():
    """Example tailored for startup recruitment."""
    print("=== Startup Recruitment Example ===")
    
    # Startup-specific configuration
    startup_config = create_custom_config(
        culture_templates={
            "startup": {
                "values": ["Innovation", "Fast-paced", "Risk-taking", "Flexibility"],
                "work_style": "Collaborative and dynamic",
                "growth_opportunities": "High"
            }
        },
        custom_params={
            "match_score_threshold": 65,  # Lower threshold for startups
            "culture_fit_threshold": 80,  # Higher culture fit requirement
            "career_growth_threshold": 70
        }
    )
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=startup_config["modules"]
    )
    
    await matcher.run(
        cv_path="path/to/startup_candidate.pdf",
        jd_path="path/to/startup_job.pdf",
        company_values="Innovation, Fast-paced, Risk-taking, Flexibility",
        industry="Technology",
        role="Full Stack Developer",
        location="Remote"
    )


async def enterprise_recruitment_example():
    """Example tailored for enterprise recruitment."""
    print("=== Enterprise Recruitment Example ===")
    
    # Enterprise-specific configuration
    enterprise_config = create_custom_config(
        custom_params={
            "match_score_threshold": 85,  # Higher threshold for enterprise
            "culture_fit_threshold": 70,
            "career_growth_threshold": 40,
            "bias_detection_sensitivity": "high"
        }
    )
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=enterprise_config["modules"]
    )
    
    await matcher.run(
        cv_path="path/to/enterprise_candidate.pdf",
        jd_path="path/to/enterprise_job.pdf",
        company_values="Stability, Process-driven, Professional development",
        industry="Technology",
        role="Enterprise Software Engineer",
        location="New York"
    )


async def remote_work_analysis_example():
    """Example focused on remote work compatibility."""
    print("=== Remote Work Analysis Example ===")
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=[
            "basic_matching",
            "culture_fit",
            "career_trajectory",
            "market_intelligence"
        ]
    )
    
    # Focus on remote work aspects
    result = await matcher.run_comprehensive_analysis(
        cv_path="path/to/remote_candidate.pdf",
        jd_path="path/to/remote_job.pdf",
        company_values="Autonomy, Communication, Self-motivation",
        industry="Technology",
        role="Remote Software Engineer",
        location="Remote"
    )
    
    # Analyze remote work compatibility
    if "culture_fit" in result:
        culture_result = result["culture_fit"]
        print(f"Remote Work Compatibility Score: {culture_result.get('culture_fit_score', 'N/A')}%")
        print(f"Communication Style: {culture_result.get('communication_style_match', 'N/A')}")
        print(f"Work Style Alignment: {culture_result.get('work_style_alignment', 'N/A')}")


async def diversity_focused_example():
    """Example with enhanced bias detection and diversity focus."""
    print("=== Diversity-Focused Analysis Example ===")
    
    # Configuration with enhanced bias detection
    diversity_config = create_custom_config(
        custom_params={
            "bias_detection_sensitivity": "high",
            "fairness_score_threshold": 90
        }
    )
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=diversity_config["modules"]
    )
    
    result = await matcher.run_comprehensive_analysis(
        cv_path="path/to/diverse_candidate.pdf",
        jd_path="path/to/inclusive_job.pdf",
        company_values="Diversity, Inclusion, Equity, Belonging",
        industry="Technology",
        role="Software Engineer",
        location="Remote"
    )
    
    # Focus on bias detection results
    if "bias_detection" in result:
        bias_result = result["bias_detection"]
        print(f"Fairness Score: {bias_result.get('fairness_score', 'N/A')}%")
        
        if bias_result.get('bias_mitigation_recommendations'):
            print("Bias Mitigation Recommendations:")
            for rec in bias_result['bias_mitigation_recommendations']:
                print(f"  • {rec}")


async def career_development_example():
    """Example focused on career development and growth."""
    print("=== Career Development Analysis Example ===")
    
    matcher = EnhancedCVJDMatcher(
        api_key="your_api_key_here",
        enabled_modules=[
            "basic_matching",
            "career_trajectory",
            "resume_optimization",
            "interview_preparation"
        ]
    )
    
    result = await matcher.run_comprehensive_analysis(
        cv_path="path/to/career_candidate.pdf",
        jd_path="path/to/growth_job.pdf",
        industry="Technology",
        role="Senior Developer",
        location="Remote"
    )
    
    # Focus on career development insights
    if "career_trajectory" in result:
        career_result = result["career_trajectory"]
        print(f"Career Growth Potential: {career_result.get('career_growth_potential', 'N/A')}%")
        print(f"Predicted Next Roles: {career_result.get('predicted_next_roles', 'N/A')}")
        print(f"Industry Relevance: {career_result.get('industry_relevance_score', 'N/A')}%")
    
    if "resume_optimization" in result:
        resume_result = result["resume_optimization"]
        print(f"Resume Optimization Score: {resume_result.get('overall_score_improvement', 'N/A')}%")


def main():
    """Run all example scenarios."""
    examples = [
        basic_usage_example,
        custom_modules_example,
        industry_specific_example,
        batch_analysis_example,
        custom_analysis_example,
        startup_recruitment_example,
        enterprise_recruitment_example,
        remote_work_analysis_example,
        diversity_focused_example,
        career_development_example
    ]
    
    print("Enhanced JD vs CV Matching System - Example Usage")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Running {example.__name__}...")
        try:
            asyncio.run(example())
        except Exception as e:
            print(f"Error in {example.__name__}: {str(e)}")
        print("-" * 40)


if __name__ == "__main__":
    main() 