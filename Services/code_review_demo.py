#!/usr/bin/env python3
"""
Intelligent Code Review and Security Analysis Demo
=================================================

This demo showcases a unique and practical use case of pydantic-ai for:
- Security vulnerability analysis in code
- Performance optimization assessment
- Code quality and best practices review
- Dependency security analysis

The system provides comprehensive code analysis with:
- Automated security vulnerability detection
- Performance bottleneck identification
- Code quality assessment
- Dependency risk analysis
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any, List

from IntelligentCodeReviewer import (
    IntelligentCodeReviewer,
    CodeLanguage,
    SecurityLevel,
    IssueCategory
)

from imageextractor.Common.constants import API_KEY


class CodeReviewDemo:
    """Demo class to showcase the Intelligent Code Reviewer."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.reviewer = IntelligentCodeReviewer(
            api_key=api_key,
            enabled_modules=[
                "security_analysis",
                "performance_analysis",
                "code_quality_analysis",
                "dependency_analysis"
            ]
        )
    
    async def demo_security_vulnerability_analysis(self):
        """Demo security vulnerability analysis."""
        print("\n" + "="*60)
        print("🔒 SECURITY VULNERABILITY ANALYSIS DEMO")
        print("="*60)
        
        # Sample vulnerable code
        vulnerable_code = """
import subprocess
import os
import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_command():
    command = request.form.get('command')
    # VULNERABLE: Command injection
    result = subprocess.check_output(command, shell=True)
    return result

@app.route('/search', methods=['GET'])
def search_users():
    query = request.args.get('q')
    # VULNERABLE: SQL injection
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name LIKE '%{query}%'")
    users = cursor.fetchall()
    return str(users)

@app.route('/template', methods=['GET'])
def render_template():
    user_input = request.args.get('user_input')
    # VULNERABLE: XSS
    template = f"<h1>Hello {user_input}!</h1>"
    return render_template_string(template)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    # VULNERABLE: Path traversal
    filename = file.filename
    file.save(os.path.join('/uploads', filename))
    return "File uploaded"

if __name__ == '__main__':
    app.run(debug=True)
        """
        
        # Create temporary file for demo
        temp_file = "temp_vulnerable_code.py"
        with open(temp_file, "w") as f:
            f.write(vulnerable_code)
        
        try:
            # Run analysis
            analysis = await self.reviewer.analyze_code(
                file_path=temp_file,
                framework="Flask",
                dependencies=["flask", "sqlite3"],
                performance_requirements="High",
                target_platform="Web Application"
            )
            
            # Print results
            self.reviewer.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Security analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_performance_analysis(self):
        """Demo performance analysis."""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE ANALYSIS DEMO")
        print("="*60)
        
        # Sample inefficient code
        inefficient_code = """
import time
import requests
from typing import List

def fetch_user_data(user_ids: List[int]) -> List[dict]:
    # INEFFICIENT: Sequential API calls
    user_data = []
    for user_id in user_ids:
        response = requests.get(f"https://api.example.com/users/{user_id}")
        user_data.append(response.json())
    return user_data

def process_large_dataset(data: List[dict]) -> List[dict]:
    # INEFFICIENT: O(n²) algorithm
    result = []
    for i, item in enumerate(data):
        for j, other_item in enumerate(data):
            if i != j and item['value'] == other_item['value']:
                result.append(item)
    return result

def fibonacci(n: int) -> int:
    # INEFFICIENT: Recursive without memoization
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def load_file_content(file_path: str) -> str:
    # INEFFICIENT: Loading entire file into memory
    with open(file_path, 'r') as f:
        return f.read()

def search_in_list(items: List[str], target: str) -> bool:
    # INEFFICIENT: Linear search for large datasets
    for item in items:
        if item == target:
            return True
    return False

# Memory leak example
class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def process_data(self, data: dict):
        # MEMORY LEAK: Never clearing cache
        key = hash(str(data))
        if key not in self.cache:
            self.cache[key] = data
        return self.cache[key]
        """
        
        # Create temporary file for demo
        temp_file = "temp_inefficient_code.py"
        with open(temp_file, "w") as f:
            f.write(inefficient_code)
        
        try:
            # Run analysis
            analysis = await self.reviewer.analyze_code(
                file_path=temp_file,
                framework="Standard Library",
                dependencies=["requests"],
                performance_requirements="High",
                target_platform="Data Processing"
            )
            
            # Print results
            self.reviewer.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Performance analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_code_quality_analysis(self):
        """Demo code quality analysis."""
        print("\n" + "="*60)
        print("📝 CODE QUALITY ANALYSIS DEMO")
        print("="*60)
        
        # Sample poor quality code
        poor_quality_code = """
import os,sys,time
from typing import *

def process_data(data):
    # POOR QUALITY: No type hints, unclear variable names
    x = []
    for i in data:
        if i > 10:
            x.append(i)
    return x

class DataHandler:
    def __init__(self):
        self.data = None
    
    def set_data(self, d):
        # POOR QUALITY: No validation, unclear parameter name
        self.data = d
    
    def get_data(self):
        # POOR QUALITY: No error handling
        return self.data
    
    def process(self):
        # POOR QUALITY: Magic numbers, no comments
        result = []
        for item in self.data:
            if item % 2 == 0:
                result.append(item * 2)
        return result

def calculate_something(a, b, c, d, e, f, g, h, i, j):
    # POOR QUALITY: Too many parameters
    return a + b + c + d + e + f + g + h + i + j

def do_something():
    # POOR QUALITY: Long function, multiple responsibilities
    data = []
    for i in range(100):
        if i % 2 == 0:
            data.append(i)
    
    result = []
    for item in data:
        if item > 50:
            result.append(item * 2)
    
    final_result = []
    for item in result:
        if item < 200:
            final_result.append(item)
    
    return final_result

# Global variable abuse
global_var = 42

def use_global():
    global global_var
    global_var += 1
    return global_var
        """
        
        # Create temporary file for demo
        temp_file = "temp_poor_quality_code.py"
        with open(temp_file, "w") as f:
            f.write(poor_quality_code)
        
        try:
            # Run analysis
            analysis = await self.reviewer.analyze_code(
                file_path=temp_file,
                framework="Standard Library",
                dependencies=[],
                coding_standards="PEP 8",
                project_type="Data Processing"
            )
            
            # Print results
            self.reviewer.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Code quality analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_dependency_analysis(self):
        """Demo dependency analysis."""
        print("\n" + "="*60)
        print("📦 DEPENDENCY ANALYSIS DEMO")
        print("="*60)
        
        # Sample code with dependencies
        code_with_dependencies = """
import requests
import numpy as np
import pandas as pd
from flask import Flask
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

def process_data():
    # Using multiple dependencies
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = data.mean()
    return result

def ml_prediction():
    # Using ML dependencies
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model
        """
        
        # Create temporary file for demo
        temp_file = "temp_dependencies_code.py"
        with open(temp_file, "w") as f:
            f.write(code_with_dependencies)
        
        try:
            # Run analysis
            analysis = await self.reviewer.analyze_code(
                file_path=temp_file,
                framework="Flask",
                dependencies=["requests", "numpy", "pandas", "flask", "tensorflow", "opencv-python", "matplotlib"],
                package_manager="pip"
            )
            
            # Print results
            self.reviewer.print_analysis_results(analysis)
            
        except Exception as e:
            print(f"❌ Dependency analysis failed: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    async def demo_custom_analysis_module(self):
        """Demo custom analysis module creation."""
        print("\n" + "="*60)
        print("🔧 CUSTOM ANALYSIS MODULE DEMO")
        print("="*60)
        
        from IntelligentCodeReviewer import AnalysisModule
        
        class CustomAccessibilityModule(AnalysisModule):
            """Custom module for accessibility analysis."""
            
            def __init__(self, llm_model):
                self.llm = llm_model
                self.agent = Agent(
                    model=self.llm,
                    system_prompt="""
                    You are an expert accessibility analyst specializing in:
                    - Web accessibility compliance (WCAG 2.1)
                    - Screen reader compatibility
                    - Keyboard navigation support
                    - Color contrast requirements
                    - Alternative text for images
                    - ARIA attributes usage
                    - Mobile accessibility
                    """,
                    retries=2,
                )
            
            async def analyze(self, code_content: str, language: CodeLanguage, **kwargs) -> Dict[str, Any]:
                prompt = f"""
                Analyze the following {language.value} code for accessibility issues:
                
                Code Content:
                {code_content}
                
                Provide accessibility analysis in JSON format with:
                - accessibility_issues (list of accessibility problems)
                - wcag_violations (WCAG 2.1 compliance issues)
                - screen_reader_issues (screen reader compatibility)
                - keyboard_navigation_issues (keyboard navigation problems)
                - color_contrast_issues (color contrast problems)
                - aria_issues (ARIA attribute problems)
                - mobile_accessibility_issues (mobile accessibility problems)
                - accessibility_recommendations (accessibility improvements)
                - accessibility_score (overall accessibility score 0-100)
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
                    return {"error": f"Accessibility analysis failed: {str(e)}"}
            
            def get_name(self) -> str:
                return "accessibility_analysis"
        
        # Add custom module to reviewer
        custom_module = CustomAccessibilityModule(self.reviewer.llm)
        self.reviewer.add_module(custom_module)
        
        print("✅ Custom accessibility analysis module added successfully!")
        print(f"Available modules: {', '.join(self.reviewer.enabled_modules)}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive code review demo."""
        print("🚀 INTELLIGENT CODE REVIEWER DEMO")
        print("="*60)
        print("This demo showcases a unique and practical use case of pydantic-ai")
        print("for automated code security, performance, and quality analysis.")
        print("="*60)
        
        # Run all demos
        await self.demo_security_vulnerability_analysis()
        await self.demo_performance_analysis()
        await self.demo_code_quality_analysis()
        await self.demo_dependency_analysis()
        await self.demo_custom_analysis_module()
        
        # Show analysis history
        print("\n" + "="*60)
        print("📊 ANALYSIS HISTORY")
        print("="*60)
        
        history = self.reviewer.get_analysis_history()
        for i, entry in enumerate(history, 1):
            print(f"\n{i}. Analysis Entry:")
            print(f"   Timestamp: {entry['timestamp']}")
            print(f"   File: {entry['file_path']}")
            print(f"   Language: {entry['language']}")
            print(f"   Modules Used: {', '.join(entry['enabled_modules'])}")
        
        print(f"\n✅ Demo completed successfully! Total analyses: {len(history)}")


async def main():
    """Main function to run the code review demo."""
    api_key = API_KEY
    
    if not api_key:
        print("❌ API key not found. Please set your API key in constants.py")
        return
    
    # Initialize and run demo
    demo = CodeReviewDemo(api_key)
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main()) 