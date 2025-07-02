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
import numpy as np
from typing import Dict, Any, Optional

class VideoAnalyzer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API_KEY is missing. Please set it in Constants.py")

        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.multimodal_agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        """Initialize the Gemini agent with required configurations."""
        return Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=self.api_key),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def _process_frame(self, frame: np.ndarray, frame_number: int, output_folder: str) -> Dict[str, Any]:
        """Process a single frame and return its analysis."""
        try:

            screenshot_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(screenshot_path, frame)

            processed_image = upload_file(screenshot_path)

            attempts = 0
            while processed_image.state.name == "PROCESSING" and attempts < 3:
                time.sleep(0.5)
                processed_image = get_file(processed_image.name)
                attempts += 1

            analysis_prompt = UI_PROMPT
            response = self.multimodal_agent.run(analysis_prompt, images=[processed_image])

            return {
                'frame_number': frame_number,
                'path': screenshot_path,
                'analysis': response.content
            }

        except Exception as error:
            return {
                'frame_number': frame_number,
                'path': screenshot_path,
                'analysis': f"Error analyzing frame {frame_number}: {error}"
            }

    def _setup_video_capture(self, video_path: str) -> Optional[tuple]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            return cap, frame_rate, total_frames
        except Exception as e:
            print(f"Error setting up video capture: {e}")
            return None

    def _ensure_output_directory(self, output_folder: str) -> bool:
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            return True
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return False

    def analyze_video(self, video_path: str, output_folder: str, interval: int = 3) -> list:


        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not self._ensure_output_directory(output_folder):
            raise RuntimeError("Failed to create output directory")

        capture_result = self._setup_video_capture(video_path)
        if not capture_result:
            raise RuntimeError("Failed to setup video capture")

        cap, frame_rate, total_frames = capture_result
        frame_interval = int(frame_rate * interval)

        print(f"Starting analysis of video with {total_frames} total frames...")
        print(f"Taking screenshots every {interval} seconds ({frame_interval} frames)")

        frame_count = 0
        results = []

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
                    print(result['analysis'])
                    print("-" * 50)

                frame_count += 1

        except Exception as e:
            print(f"Error during video analysis: {e}")
        finally:
            cap.release()
            print("\nVideo analysis complete!")

        return results

    def generate_report(self, results: list, output_folder: str) -> None:
        """Generate a summary report of the analysis."""
        try:
            report_path = os.path.join(output_folder, 'analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write("Video Analysis Summary Report\n")
                f.write("=" * 30 + "\n\n")

                for result in results:
                    f.write(f"Frame {result['frame_number']}:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Screenshot: {result['path']}\n")
                    f.write(f"Analysis:\n{result['analysis']}\n\n")

            print(f"Report generated successfully: {report_path}")
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