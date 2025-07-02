import cv2
import os
import time
from pathlib import Path
import tempfile
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
from Common.constants import *
from typing import Dict, Any, Optional
import re
import numpy as np

class TextAnalyzer:

    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """Analyze text for grammar, clarity, and consistency."""
        analysis_prompt = """
        Analyze the following text and provide specific corrections for:
        1. Grammar and spelling errors
        2. Inconsistent capitalization
        3. Technical jargon that needs simplification
        4. Unclear or ambiguous phrases
        5. Punctuation issues

        Provide the corrected version of the text along with specific improvements made.

        Text to analyze:
        {text}
        """

        text_agent = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=API_KEY),
            tools=[],
            markdown=True,
        )

        response = text_agent.run(analysis_prompt.format(text=text))
        return {
            'original_text': text,
            'analysis': response.content,
            'corrections': TextAnalyzer._extract_corrections(response.content)
        }

    @staticmethod
    def _extract_corrections(analysis: str) -> Dict[str, list]:
        """Extract specific corrections from the analysis."""
        corrections = {
            'grammar': [],
            'capitalization': [],
            'clarity': [],
            'jargon': [],
            'punctuation': []
        }

        current_category = None
        for line in analysis.split('\n'):
            line = line.strip()
            for category in corrections.keys():
                if category.title() in line:
                    current_category = category
                    continue
            if current_category and line.startswith('-'):
                corrections[current_category].append(line[1:].strip())

        return corrections

class VideoAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the VideoAnalyzer with necessary configurations."""
        if not api_key:
            raise ValueError("API_KEY is missing. Please set it in Constants.py")

        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.multimodal_agent = self._initialize_agent()
        self.text_analyzer = TextAnalyzer()

    def _initialize_agent(self) -> Agent:
        """Initialize the Gemini agent with required configurations."""
        return Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=self.api_key),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def _process_frame(self, frame: np.ndarray, frame_number: int, output_folder: str) -> Dict[str, Any]:
        """Process a single frame and return its analysis with text review."""
        try:
            screenshot_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(screenshot_path, frame)

            processed_image = upload_file(screenshot_path)

            attempts = 0
            while processed_image.state.name == "PROCESSING" and attempts < 3:
                time.sleep(0.5)
                processed_image = get_file(processed_image.name)
                attempts += 1

            analysis_prompt = """
            Analyze this screenshot in detail:
            1. UI Elements and Layout
            2. Text Content Quality:
               - Identify any text present
               - Check grammar and spelling
               - Assess clarity and readability
               - Note any technical jargon
            3. Process Flow
            4. Compliance Requirements

            Provide detailed analysis with corrected text where applicable.
            """

            response = self.multimodal_agent.run(analysis_prompt, images=[processed_image])

            text_analysis = self.text_analyzer.analyze_text(response.content)

            return {
                'frame_number': frame_number,
                'path': screenshot_path,
                'original_analysis': response.content,
                'text_review': text_analysis,
                'final_analysis': text_analysis['corrections']
            }

        except Exception as error:
            return {
                'frame_number': frame_number,
                'path': screenshot_path,
                'error': f"Error analyzing frame {frame_number}: {error}"
            }

    def analyze_video(self, video_path: str, output_folder: str, interval: int = 3) -> list:
        """Analyze video frames with text quality review."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate * interval)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Starting analysis of video with {total_frames} total frames...")
        print(f"Taking screenshots every {interval} seconds ({frame_interval} frames)")

        results = []
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    print(f"\nProcessing frame {frame_count}/{total_frames}")
                    result = self._process_frame(frame, frame_count, output_folder)
                    results.append(result)

                    print(f"\nAnalysis for frame {result['frame_number']}:")
                    print("-" * 50)

                    if 'error' in result:
                        print(result['error'])
                    else:
                        print("Original Analysis:")
                        print(result['original_analysis'])
                        print("\nText Review and Corrections:")
                        for category, corrections in result['final_analysis'].items():
                            if corrections:
                                print(f"\n{category.title()} Corrections:")
                                for correction in corrections:
                                    print(f"- {correction}")

                    print("-" * 50)

                frame_count += 1

        finally:
            cap.release()
            print("\nVideo analysis complete!")

        return results

    def generate_report(self, results: list, output_folder: str) -> None:
        """Generate a detailed report including text analysis."""
        try:
            report_path = os.path.join(output_folder, 'analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write("Video Analysis Summary Report\n")
                f.write("=" * 30 + "\n\n")

                for result in results:
                    f.write(f"Frame {result['frame_number']}:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Screenshot: {result['path']}\n\n")

                    if 'error' in result:
                        f.write(f"Error: {result['error']}\n")
                    else:
                        f.write("Original Analysis:\n")
                        f.write(result['original_analysis'] + "\n\n")

                        f.write("Text Review and Corrections:\n")
                        for category, corrections in result['final_analysis'].items():
                            if corrections:
                                f.write(f"\n{category.title()} Corrections:\n")
                                for correction in corrections:
                                    f.write(f"- {correction}\n")

                    f.write("\n" + "=" * 30 + "\n\n")

            print(f"Detailed report generated: {report_path}")
        except Exception as e:
            print(f"Error generating report: {e}")

def main():
    video_file_path = "D:\\demo\\paynow.mp4"
    output_screenshot_folder = "D:\\demo\\screenshots"

    try:
        analyzer = VideoAnalyzer(API_KEY)

        if video_file_path:
            results = analyzer.analyze_video(
                video_path=video_file_path,
                output_folder=output_screenshot_folder
            )

            analyzer.generate_report(results, output_screenshot_folder)
        else:
            print("Please provide a valid video file.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()