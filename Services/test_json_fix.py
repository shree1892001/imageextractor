#!/usr/bin/env python3
"""
Test script to verify JSON fixing functionality.
"""

import json
import re
from imageextractor.Services.JDVsCV_Enhanced import AnalysisModule

class TestAnalysisModule(AnalysisModule):
    """Test class to access the _fix_json_string method."""
    
    async def analyze(self, cv_text: str, jd_text: str, **kwargs) -> Dict[str, Any]:
        return {}
    
    def get_name(self) -> str:
        return "test"

def test_json_fixing():
    """Test the JSON fixing functionality."""
    test_module = TestAnalysisModule(None)
    
    # Test case 1: The problematic JSON from the error
    problematic_json = '''{
        "keyword_optimization": {
            "description": "The resume needs significant keyword optimization to align with the target job description for a React Developer. Currently, it focuses on Python skills, which are irrelevant. We need to inject React-specific keywords throughout.",
            "actions": [
                "Add keywords like \\'React.js\\', \\'Redux\\', \\'React Hooks\\', \\'Context API\\', \\'JSX\\', \\'Component lifecycle\\', \\'state management\\', \\'props\\', \\'functional components\\', \\'class components\\', \\'virtual DOM\\', \\'fiber architecture\\', \\'SPA (Single Page Application)\\', \\'React Router\\', \\'testing libraries (Jest, React Testing Library)\\', \\'Webpack\\', \\'Babel\\' throughout the resume, especially in the experience section.",
                "Quantify achievements with numbers wherever possible (e.g., \\'Improved app performance by 15%\\').",
                "Replace generic descriptions with action verbs and quantifiable results."
            ]
        },
        "section_reordering": {
            "description": "The current section order is confusing and doesn\\'t follow standard resume best practices. A more impactful order is needed.",
            "actions": [
                "Move \\'Professional Summary\\' to the top.",
                "Rename \\'TECH STACK\\' to \\'Skills\\' and move it after \\'Professional Summary\\'.",
                "Rename \\'EMPLOYMENT HISTORY\\' to \\'Experience\\' and move it after \\'Skills\\'.",
                "Move \\'Education\\' after \\'Experience\\'.",
                "Add a \\'Projects\\' section if relevant projects exist (This section is missing entirely in the provided resume).",
                "Consider adding a \\'Certifications\\' section if applicable."
            ]
        },
        "content_enhancements": {
            "description": "The content is weak and lacks impact. It needs to showcase achievements and quantify results.",
            "actions": [
                "Rewrite the \\'About\\' section to focus on React skills and accomplishments.",
                "Expand on the bullet points under \\'Experience\\' with specific accomplishments and quantifiable results (e.g., improved performance by X%, reduced bugs by Y%).",
                "Add details about the technologies used in each project, highlighting the role of React.",
                "Use the STAR method (Situation, Task, Action, Result) to describe accomplishments in detail."
            ]
        },
        "skill_highlighting": {
            "description": "Skills are mentioned but not effectively highlighted. They need to be categorized and formatted for better readability and ATS compatibility.",
            "actions": [
                "Create a dedicated \\'Skills\\' section with clear categories (e.g., Front-End Frameworks, Languages, Testing, Databases, Tools).",
                "Use a consistent format for skills (e.g., bullet points or a table).",
                "Ensure keywords are included in both the skills section and throughout the resume body."
            ]
        },
        "experience_reframing": {
            "description": "The experience section is poorly structured and lacks impact. It needs to be rewritten to highlight accomplishments.",
            "actions": [
                "Rewrite each experience entry using the STAR method, focusing on quantifiable achievements and results.",
                "Use action verbs to start each bullet point.",
                "Focus on the impact of each accomplishment on the business or team.",
                "Remove generic statements and replace them with specific examples."
            ]
        },
        "overall_score_improvement": {
            "description": "The resume currently scores very low due to its lack of relevance to the target job description and poor structure. Significant improvements are needed.",
            "actions": [
                "Completely rewrite the resume to focus on React development skills and experience.",
                "Use a professional resume template.",
                "Tailor the resume to each specific job application.",
                "Proofread carefully for grammar and spelling errors."
            ],
            "estimated_improvement": "70-80%"
        },
        "specific_edits": {
            "description": "Example edits to illustrate the necessary changes.",
            "examples": [
                {
                    "original": "Demonstrated proficiency in using Python to develop scalable and efficient software applications.",
                    "revised": "Developed and deployed high-performance React applications, leveraging advanced state management techniques (Redux, Context API) to enhance user experience and improve overall application efficiency by 15%."
                },
                {
                    "original": "Experienced in using frameworks such as Django and Flask for web development.",
                    "revised": "Expert in building scalable and maintainable React applications, proficient in utilizing React Router for seamless navigation and React Hooks for efficient state management."
                },
                {
                    "original": "Currently, I am part of an agile team responsible for building dynamic web applications using Flask.",
                    "revised": "Currently leading the front-end development of a dynamic web application using React.js, Redux, and other modern technologies. Successfully implemented a new feature resulting in a 20% increase in user engagement."
                }
            ]
        }
    }'''
    
    print("Testing JSON fixing...")
    print("Original JSON length:", len(problematic_json))
    
    try:
        # Try to parse the original JSON
        original_parsed = json.loads(problematic_json)
        print("✅ Original JSON is valid!")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Original JSON has error: {e}")
        print(f"Error position: {e.pos}")
        
        # Show the problematic area
        start_pos = max(0, e.pos - 50)
        end_pos = min(len(problematic_json), e.pos + 50)
        print(f"Problematic area: ...{problematic_json[start_pos:e.pos]}>>>ERROR<<<{problematic_json[e.pos:end_pos]}...")
        
        # Try to fix the JSON
        fixed_json = test_module._fix_json_string(problematic_json)
        print(f"Fixed JSON length: {len(fixed_json)}")
        
        try:
            fixed_parsed = json.loads(fixed_json)
            print("✅ Fixed JSON is valid!")
            return True
        except json.JSONDecodeError as e2:
            print(f"❌ Fixed JSON still has error: {e2}")
            print(f"Error position: {e2.pos}")
            
            # Show the problematic area in fixed JSON
            start_pos = max(0, e2.pos - 50)
            end_pos = min(len(fixed_json), e2.pos + 50)
            print(f"Fixed JSON problematic area: ...{fixed_json[start_pos:e2.pos]}>>>ERROR<<<{fixed_json[e2.pos:end_pos]}...")
            return False

if __name__ == "__main__":
    test_json_fixing() 