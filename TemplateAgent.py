import cv2
import os
import re
import json
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from typing import Dict, Any
from Common.constants import API_KEY

class WebCodeFormatter:
    def __init__(self, api_key: str):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API_KEY is required")

        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.multimodal_agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        return Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=self.api_key),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def analyze_and_format(self, screenshot_path: str, code_file_path: str, output_dir: str) -> bool:
        try:
            ui_patterns = self.extract_ui_patterns(screenshot_path)
            if not ui_patterns:
                raise ValueError("Failed to extract UI patterns from screenshot")

            with open(code_file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()

            code_info = self.analyze_code_type(code_content, code_file_path)
            formatted_code = self.format_code(code_content, code_info, ui_patterns)
            return self.save_formatted_code(formatted_code, output_dir)

        except Exception as error:
            print(f"Error in analyze_and_format: {str(error)}")
            return False

    def analyze_code_type(self, code_content: str, file_path: str) -> Dict[str, Any]:
        code_info = {'type': None}
        file_ext = os.path.splitext(file_path)[-1].lower()

        if file_ext == ".html" or "<html" in code_content:
            code_info['type'] = 'html'
        elif file_ext == ".css" or ('{' in code_content and '}' in code_content and ':' in code_content):
            code_info['type'] = 'css'
        elif file_ext == ".js" or 'function' in code_content:
            code_info['type'] = 'javascript'
        else:
            raise ValueError(f"Unable to determine code type for file: {file_path}")

        print(f"✅ Detected code type: {code_info['type']}")
        return code_info

    def extract_ui_patterns(self, screenshot_path: str) -> Dict[str, Any]:
        try:
            image = cv2.imread(screenshot_path)
            if image is None:
                raise ValueError(f"Failed to load screenshot from {screenshot_path}")

            processed_image = genai.upload_file(screenshot_path)
            patterns_prompt = """
            Analyze this UI screenshot and extract patterns as JSON:
            {
              "html": "<html_structure>",
              "css": {"header_footer_color": "#d0e7ff"},
              "javascript": "<js_logic>"
            }
            """

            response = self.multimodal_agent.run(patterns_prompt, images=[processed_image])
            json_data = self._extract_json(response.content)

            if not json_data:
                raise ValueError("AI response is not a valid JSON.")

            return json_data
        except Exception as error:
            print(f"Error extracting patterns: {str(error)}")
            return {}

    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {}
        except json.JSONDecodeError:
            return {}

    def format_code(self, code_content: str, code_info: Dict[str, Any], ui_patterns: Dict[str, Any]) -> str:
        try:
            if code_info['type'] != 'html':
                return code_content

            # Remove predefined header and footer divs
            filtered_content = re.sub(r'<div[^>]*header[^>]*>.*?</div>', '', code_content, flags=re.DOTALL | re.IGNORECASE)
            filtered_content = re.sub(r'<div[^>]*footer[^>]*>.*?</div>', '', filtered_content, flags=re.DOTALL | re.IGNORECASE)

            formatted_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Responsive Page</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}

                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f8f8f8;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                    }}

                    .container {{
                        width: 90%;
                        max-width: 1200px;
                        border: 3px solid lightblue;
                        background-color: white;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                        border-radius: 10px;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        text-align: center;
                    }}

                    header, footer {{
                        background-color: {ui_patterns.get('css', {}).get('header_footer_color', '#d0e7ff')};
                        padding: 15px;
                        font-size: 20px;
                        font-weight: bold;
                    }}

                    main {{
                        padding: 20px;
                        flex: 1;
                        text-align: left;
                    }}

                    /* Responsive Design */
                    @media (max-width: 768px) {{
                        .container {{
                            width: 95%;
                        }}
                        header, footer {{
                            font-size: 18px;
                            padding: 10px;
                        }}
                        main {{
                            padding: 15px;
                        }}
                    }}

                    @media (max-width: 480px) {{
                        .container {{
                            width: 100%;
                            border: none;
                        }}
                        header, footer {{
                            font-size: 16px;
                            padding: 8px;
                        }}
                        main {{
                            padding: 10px;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    
                    <main>
                        {filtered_content}
                    </main>
                   
                </div>
            </body>
            </html>
            """

            return formatted_html
        except Exception as error:
            print(f"Error formatting code: {str(error)}")
            return code_content

    def save_formatted_code(self, formatted_code: str, output_dir: str) -> bool:
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "formatted.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_code)
            print(f"✅ Formatted HTML saved to: {output_path}")
            return True
        except Exception as error:
            print(f"Error saving formatted code: {str(error)}")
            return False

if __name__ == "__main__":
    formatter = WebCodeFormatter(API_KEY)
    success = formatter.analyze_and_format("D:\\demo\\image.png", "D:\\demo\\pay_later.html",
                                           "D:\\demo\\paylater1.html")
    if success:
        print("✅ Code formatting completed successfully")
    else:
        print("❌ Failed to format code")
