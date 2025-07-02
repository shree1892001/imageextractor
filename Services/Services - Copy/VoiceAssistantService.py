import os
import re
import json
import PyPDF2
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from datetime import datetime
import time
import threading
from langdetect import detect, DetectorFactory
import pycountry
from googletrans import Translator
import warnings
from Common.constants import *

# Suppress warnings
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0


class MultilingualAssistant:
    def __init__(self):
        # Configuration
        self.api_key = API_KEY_5
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.transcript_file = f"assistant_log_{self.session_id}.txt"
        self.voice_transcript_file = f"voice_transcript_{self.session_id}.txt"
        self.conversation_history = []
        self.voice_transcript = []

        self.default_language = "en"
        self.current_language = self.default_language
        self.translator = Translator()

        self.job_description = ""
        self.screening_questions = []
        self.current_question_index = 0
        self.candidate_responses = []
        self.interview_feedback = ""

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.init_tts_engine()

        self.init_ai_model()

        self.listening = True
        self.active_mode = False

    def init_tts_engine(self):
        """Initialize text-to-speech engine with more voice options"""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')

            # Configure voice properties
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)

            for voice in voices:
                if "natural" in voice.name.lower() or "female" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            self.speak("Assistant initialized successfully")
        except Exception as e:
            print(f"TTS initialization error: {e}")
            self.engine = None

    def init_ai_model(self):
        """Initialize the AI model with enhanced configuration"""
        genai.configure(api_key=self.api_key)

        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        try:
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            self.convo = self.model.start_chat(history=[])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI model: {e}")

    def speak(self, text, wait=True):
        """Speak text using TTS and log to voice transcript with text cleaning"""

        import re

        cleaned_text = re.sub(r'\*+', ' ', text)

        cleaned_text = re.sub(r'[^\w\s.,;:?!()$%#@-]', ' ', cleaned_text)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        print(f"Assistant: {cleaned_text}")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.voice_transcript.append(f"[{timestamp}] Assistant: {cleaned_text}\n")

        self.save_voice_transcript()

        if self.engine:
            try:
                self.engine.say(cleaned_text)
                if wait:
                    self.engine.runAndWait()
            except Exception as e:
                print(f"Error in TTS: {e}")

    def save_voice_transcript(self):
        """Save the voice transcript to file"""
        try:
            with open(self.voice_transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Voice Assistant Transcript - Session {self.session_id}\n\n")
                f.write("".join(self.voice_transcript))
        except Exception as e:
            print(f"Error saving voice transcript: {e}")

    def save_transcript(self):
        """Save the complete conversation transcript"""
        try:
            with open(self.transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"Assistant Transcript - Session {self.session_id}\n\n")
                f.write(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                if self.job_description:
                    f.write("=== JOB SCREENING DETAILS ===\n")
                    f.write(f"Job Description:\n{self.job_description}\n\n")
                    f.write("Generated Questions:\n")
                    for i, q in enumerate(self.screening_questions, 1):
                        f.write(f"{i}. [{q['category']}] {q['question']}\n")
                        f.write(f"   Evaluation: {q['evaluation']}\n\n")

                    f.write("\n=== INTERVIEW RESPONSES ===\n")
                    for response in self.candidate_responses:
                        f.write(f"Question: {response['question']}\n")
                        f.write(f"Response: {response['response']}\n")
                        f.write(f"Evaluation: {response['evaluation']}\n\n")

                    f.write("\n=== FINAL FEEDBACk ===")
                    f.write(self.interview_feedback + "\n\n")

                f.write("\n=== CONVERSATION HISTORY ===\n")
                for item in self.conversation_history:
                    f.write(f"{item['role'].upper()}: {item['content']}\n")

                f.write("\n=== VOICE TRANSCRIPT ===\n")
                f.write("".join(self.voice_transcript))

            print(f"Transcript saved to {self.transcript_file}")
            print(f"Voice transcript saved to {self.voice_transcript_file}")
        except Exception as e:
            print(f"Error saving transcript: {e}")

    def listen(self, timeout=10):
        """Enhanced speech recognition with better handling of low-volume speech"""
        with self.microphone as source:
            print("\nListening... (Speak now)")

            try:

                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                self.recognizer.energy_threshold = 4000

                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.dynamic_energy_adjustment_damping = 0.15
                self.recognizer.dynamic_energy_adjustment_ratio = 1.5

                self.recognizer.pause_threshold = 0.8

                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=20
                )

                text = self.recognizer.recognize_google(
                    audio,
                    language=self.current_language,
                    show_all=False
                )

                print(f"User: {text}")

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.voice_transcript.append(f"[{timestamp}] User: {text}\n")
                self.save_voice_transcript()

                return text

            except sr.WaitTimeoutError:
                print("Listening timed out (no speech detected)")
                return None

            except sr.UnknownValueError:
                print("Could not understand audio (possibly too quiet)")
                self.speak("Sorry, I didn't catch that. Could you speak a bit louder?")
                return None

            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None

            except Exception as e:
                print(f"Unexpected error in listening: {e}")
                return None

    def read_text_file(self, file_path):
        """Read text from various file formats"""
        try:
            if file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_path.lower().endswith('.docx'):
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    def startup_prescreening_mode(self):
        """Specialized mode for startup pre-screening interviews"""
        self.speak("Welcome to Startup Pre-screening Mode. Let's set up the interview.")

        # Get job description
        self.speak("First, please provide the job description.")
        print("\nJob Description Options:")
        print("1. Paste text")
        print("2. Upload file (PDF, DOCX, TXT)")
        print("3. Speak description")
        print("4. Return to main menu")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("Paste the job description (Ctrl+D to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            self.job_description = "\n".join(lines)
        elif choice == "2":
            file_path = input("Enter file path: ").strip()
            try:
                self.job_description = self.read_text_file(file_path)
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        elif choice == "3":
            self.speak("Please speak the job description clearly.")
            self.job_description = self.listen(timeout=90)
        elif choice == "4":
            return
        else:
            print("Invalid choice")
            return

        if not self.job_description or len(self.job_description.strip()) < 50:
            self.speak("The description seems too short. Please try again with more details.")
            return

        # Generate startup-specific questions
        self.speak("Generating startup-specific screening questions...")
        prompt = f"""Generate 8-10 pre-screening questions specifically for a startup environment based on this job description ,make sure the questions are basic.
        Focus on:
        1. Adaptability to fast-paced environments
        2. Problem-solving in resource-constrained situations
        3. Cultural fit for startup culture
        4. Ability to wear multiple hats
        5. Handling ambiguity and rapid change
        6. Passion for the startup's mission
        7. Previous startup experience (if relevant)
        8. Risk tolerance

        For each question, include:
        - Why it's important for this role
        - What to look for in responses

        Job Description:
        {self.job_description}

        Return questions in this format:
        CATEGORY: [category]
        QUESTION: [question text]
        IMPORTANCE: [why this matters for startups]
        EVALUATION: [what to look for in answers]
        """

        try:
            response = self.convo.send_message(prompt).text
            self.screening_questions = []

            # Parse the response
            current_question = {}
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("CATEGORY:"):
                    if current_question:
                        self.screening_questions.append(current_question)
                    current_question = {
                        "category": line.replace("CATEGORY:", "").strip(),
                        "question": "",
                        "importance": "",
                        "evaluation": ""
                    }
                elif line.startswith("QUESTION:"):
                    current_question["question"] = line.replace("QUESTION:", "").strip()
                elif line.startswith("IMPORTANCE:"):
                    current_question["importance"] = line.replace("IMPORTANCE:", "").strip()
                elif line.startswith("EVALUATION:"):
                    current_question["evaluation"] = line.replace("EVALUATION:", "").strip()

            if current_question:
                self.screening_questions.append(current_question)

            print("\nGenerated Startup Screening Questions:")
            for i, q in enumerate(self.screening_questions, 1):
                print(f"{i}. [{q['category']}] {q['question']}")
                print(f"   Why this matters: {q['importance']}")
                print(f"   Evaluation: {q['evaluation']}\n")

            self.speak(
                f"I've prepared {len(self.screening_questions)} startup-specific questions. Let's begin the interview.")
            self.conduct_interview()

        except Exception as e:
            print(f"Error generating startup questions: {e}")
            self.speak("I encountered an error generating the questions. Please try again.")

    def analyze_job_description(self, jd_text):
        """Analyze job description and extract key components"""
        prompt = f"""Analyze this job description and extract:
        1. Key skills required
        2. Experience level needed
        3. Responsibilities
        4. Qualifications
        5. Any special requirements

        Return the analysis in a structured format.

        Job Description:
        {jd_text}"""

        try:
            response = self.convo.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Error analyzing job description: {e}")
            return None

    def generate_questions(self):
        """Generate more sophisticated interview questions from job description"""
        prompt = f"""Based on this job description, generate 7-10 high-quality interview questions that would effectively screen candidates.
        Organize them by category (technical, behavioral, situational).
        For each question, include what a good answer should cover.

        Job Description:
        {self.job_description}

        Return the questions in this EXACT format:
        CATEGORY: [category name]
        QUESTION: [question text]
        EVALUATION: [what to look for in answers]

        (Repeat the above three lines for each question)
        """

        try:
            response = self.convo.send_message(prompt)
            questions = []

            # Use regex to extract question blocks
            question_blocks = re.findall(r"CATEGORY:\s*(.*?)\s*QUESTION:\s*(.*?)\s*EVALUATION:\s*(.*?)(?=CATEGORY:|$)",
                                         response.text, re.DOTALL | re.IGNORECASE)

            for category, question, evaluation in question_blocks:
                questions.append({
                    "category": category.strip(),
                    "question": question.strip(),
                    "evaluation": evaluation.strip()
                })

            # If regex didn't work, try fallback parsing
            if not questions:
                current_question = {"category": "", "question": "", "evaluation": ""}
                for line in response.text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if re.match(r"^(?:CATEGORY:|Category:)\s*(.*)", line, re.IGNORECASE):
                        # If we have a previous question with content, save it
                        if current_question["question"]:
                            questions.append(current_question.copy())

                        # Start a new question
                        current_question = {"category": "", "question": "", "evaluation": ""}
                        current_question["category"] = re.match(r"^(?:CATEGORY:|Category:)\s*(.*)", line,
                                                                re.IGNORECASE).group(1).strip()

                    elif re.match(r"^(?:QUESTION:|Question:)\s*(.*)", line, re.IGNORECASE):
                        current_question["question"] = re.match(r"^(?:QUESTION:|Question:)\s*(.*)", line,
                                                                re.IGNORECASE).group(1).strip()

                    elif re.match(r"^(?:EVALUATION:|Evaluation Criteria:)\s*(.*)", line, re.IGNORECASE):
                        current_question["evaluation"] = re.match(r"^(?:EVALUATION:|Evaluation Criteria:)\s*(.*)", line,
                                                                  re.IGNORECASE).group(1).strip()

                # Add the last question if it has content
                if current_question["question"]:
                    questions.append(current_question)

            print(f"Generated {len(questions)} questions")
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def evaluate_response(self, question, response):
        """Evaluate candidate response against expected criteria"""
        prompt = f"""Evaluate this interview response based on the question and evaluation criteria.
        Provide specific feedback on what was good and what could be improved.
        Also rate the response on a scale of 1-5.

        Question: {question['question']}
        Evaluation Criteria: {question['evaluation']}
        Candidate Response: {response}

        Return your evaluation in this format:
        Rating: [1-5]
        Strengths: [list strengths]
        Areas for Improvement: [list improvements]
        Detailed Feedback: [detailed analysis]"""

        try:
            evaluation = self.convo.send_message(prompt).text
            return evaluation
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return "Could not evaluate this response."

    def conduct_interview(self):
        """Enhanced interview conduction with real-time feedback"""
        self.speak("Let's begin the interview. I'll ask you questions based on the job description.")
        self.speak(f"There will be {len(self.screening_questions)} questions in total.")

        for i, question in enumerate(self.screening_questions):
            self.speak(f"Question {i + 1} of {len(self.screening_questions)}:")
            self.speak(f"This is a {question['category']} question: {question['question']}")

            max_attempts = 2
            response = None

            for attempt in range(max_attempts):
                response = self.listen(timeout=30)
                if response:
                    break
                elif attempt < max_attempts - 1:
                    self.speak("I didn't catch that. Could you please repeat your answer?")

            if response:
                evaluation = self.evaluate_response(question, response)
                self.candidate_responses.append({
                    "question": question['question'],
                    "category": question['category'],
                    "response": response,
                    "evaluation": evaluation
                })

                # Give brief feedback
                rating_match = re.search(r"Rating:\s*([1-5])", evaluation)
                if rating_match:
                    rating = rating_match.group(1)
                    self.speak(f"Thank you. I've rated that response as {rating} out of 5.")
            else:
                self.candidate_responses.append({
                    "question": question['question'],
                    "category": question['category'],
                    "response": "No response provided",
                    "evaluation": "Could not evaluate - no response"
                })
                self.speak("Let's move to the next question.")

        # Generate overall feedback
        self.generate_feedback()
        self.speak("The interview is now complete. I'll provide your overall feedback.")
        self.speak(self.interview_feedback)
        self.save_transcript()

    def generate_feedback(self):
        """Generate comprehensive feedback on the entire interview"""
        responses_text = "\n".join(
            f"Question: {q['question']}\nResponse: {q['response']}\nEvaluation: {q['evaluation']}\n"
            for q in self.candidate_responses
        )

        prompt = f"""Based on these interview responses, provide comprehensive feedback:
        1. Overall strengths demonstrated
        2. Key areas needing improvement
        3. Suggested areas for preparation
        4. Final recommendation (Strong Yes, Yes, Maybe, No)

        Interview Responses:
        {responses_text}"""

        try:
            self.interview_feedback = self.convo.send_message(prompt).text
        except Exception as e:
            self.interview_feedback = "Could not generate comprehensive feedback."
            print(f"Error generating feedback: {e}")

    def text_to_speech_mode(self):
        """Enhanced TTS mode that can also answer questions about the content"""
        print("\n=== Text-to-Speech Mode ===")

        # Initialize content storage if it doesn't exist
        if not hasattr(self, 'current_content'):
            self.current_content = ""

        while True:
            print("1. Enter text to speak")
            print("2. Read from a file")
            print("3. Ask questions about content")
            print("4. Return to main menu")

            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                print("\nEnter the text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        if lines and not lines[-1]:
                            break
                        if not lines:
                            continue
                    lines.append(line)
                self.current_content = "\n".join(lines).strip()
                if self.current_content:
                    self.speak("I'll now read this content. Say 'stop' to pause.")
                    self.read_content_interactively(self.current_content)

            elif choice == "2":
                file_path = input("Enter file path: ").strip()
                if os.path.exists(file_path):
                    try:
                        self.current_content = self.read_text_file(file_path)
                        if self.current_content:
                            print("\nContent preview (first 200 chars):")
                            print(self.current_content[:200] + ("..." if len(self.current_content) > 200 else ""))
                            self.speak("I'll now read this content. Say 'stop' to pause.")
                            self.read_content_interactively(self.current_content)
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("File not found!")

            elif choice == "3":
                if not self.current_content:
                    print("No content loaded. Would you like to load content now?")
                    load_choice = input(
                        "1. Enter text\n2. Read from file\n3. Return to menu\nEnter choice (1-3): ").strip()

                    if load_choice == "1":
                        print("\nEnter the text (press Enter twice to finish):")
                        lines = []
                        while True:
                            line = input()
                            if not line:
                                if lines and not lines[-1]:
                                    break
                                if not lines:
                                    continue
                            lines.append(line)
                        self.current_content = "\n".join(lines).strip()

                    elif load_choice == "2":
                        file_path = input("Enter file path: ").strip()
                        if os.path.exists(file_path):
                            try:
                                self.current_content = self.read_text_file(file_path)
                                print("\nContent loaded successfully!")
                            except Exception as e:
                                print(f"Error reading file: {e}")
                                continue
                        else:
                            print("File not found!")
                            continue

                    elif load_choice == "3":
                        continue
                    else:
                        print("Invalid choice")
                        continue

                    if not self.current_content:
                        print("No content was loaded. Please try again.")
                        continue

                print("\nContent available for questions (first 100 chars):")
                print(self.current_content[:100] + ("..." if len(self.current_content) > 100 else ""))
                self.speak("You can now ask questions about the content. Say 'exit' to stop.")

                while True:
                    question = self.listen()
                    if not question:
                        continue
                    if "exit" in question.lower():
                        break

                    prompt = f"""Content: {self.current_content}
                    Question: {question}
                    Answer the question based on the content above."""

                    try:
                        response = self.convo.send_message(prompt).text
                        self.speak(response)
                    except Exception as e:
                        self.speak("Sorry, I couldn't process that question.")
                        print(f"Error: {e}")

            elif choice == "4":
                return
            else:
                print("Invalid choice")

    def read_content_interactively(self, content):
        """Read content with interactive controls"""
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            self.speak(chunk, wait=False)

            # Check for stop command
            stop_command = self.listen(timeout=2)
            if stop_command and "stop" in stop_command.lower():
                self.speak("Pausing reading.")
                action = input("Continue reading? (y/n): ").lower()
                if action != 'y':
                    break

    def text_chat_mode(self):
        """Enhanced text chat mode with more conversational flow"""
        print("\n=== Text Chat Mode ===")
        print("Type your questions or 'exit' to quit.\n")

        self.speak("I'm ready to answer your questions. Please type your question.")

        while self.listening:
            try:
                user_input = input("You: ")
                if not user_input.strip():
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting text chat mode...")
                    break

                # Get AI response with conversation context
                response = self.convo.send_message(user_input).text
                print(f"Assistant: {response}")
                self.speak(response)

                # Add to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

            except KeyboardInterrupt:
                print("\nExiting text chat mode...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def voice_assistant_mode(self):
        """Enhanced voice assistant mode with more natural interaction"""
        self.speak("Hello! I'm your voice assistant. How can I help you today?")

        while self.listening:
            try:
                user_input = self.listen(timeout=7)
                if not user_input:
                    continue

                if "exit" in user_input.lower() or "quit" in user_input.lower():
                    self.speak("Goodbye! Have a great day.")
                    break

                # Show we're processing
                self.speak("Let me think about that...", wait=True)

                # Get thoughtful response
                prompt = f"""The user asked: {user_input}
                Provide a comprehensive, helpful response in a conversational tone."""

                response = self.convo.send_message(prompt).text
                self.speak(response)

                # Add to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

            except KeyboardInterrupt:
                self.speak("Ending voice assistant mode.")
                break
            except Exception as e:
                self.speak("Sorry, I encountered an issue. Please try again.")
                print(f"Error: {e}")

    def job_screening_mode(self):
        """Enhanced job screening mode with better analysis"""
        self.speak("Let's set up a job screening interview. First, I need the job description.")
        print("\nJob Description Options:")
        print("1. Paste text")
        print("2. Upload file (PDF, DOCX, TXT)")
        print("3. Speak description")
        print("4. Return to main menu")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("Paste the job description (Ctrl+D to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            self.job_description = "\n".join(lines)
        elif choice == "2":
            file_path = input("Enter file path: ").strip()
            try:
                self.job_description = self.read_text_file(file_path)
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        elif choice == "3":
            self.speak("Please speak the job description clearly.")
            self.job_description = self.listen(timeout=90)
        elif choice == "4":
            return
        else:
            print("Invalid choice")
            return

        if not self.job_description or len(self.job_description.strip()) < 50:
            self.speak("The description seems too short. Please try again with more details.")
            return

        # Analyze the job description
        self.speak("Analyzing the job description...")
        analysis = self.analyze_job_description(self.job_description)
        print("\nJob Description Analysis:")
        print(analysis)
        self.speak("Here's what I understand about this position. I'll now generate interview questions.")

        # Generate questions
        self.screening_questions = self.generate_questions()
        if not self.screening_questions:
            self.speak("I couldn't generate questions for this description. It might be too vague.")
            return

        print("\nGenerated Questions:")
        for i, q in enumerate(self.screening_questions, 1):
            print(f"{i}. [{q['category']}] {q['question']}")
            print(f"   Evaluation: {q['evaluation']}\n")

        self.speak(f"I've prepared {len(self.screening_questions)} questions. Let's begin the interview.")
        self.conduct_interview()

    def run(self):
        """Main menu with enhanced options"""
        try:
            while True:
                print("\n=== AI Assistant ===")
                print("1. Text-to-Speech & Content Q&A")
                print("2. Text Chat Assistant")
                print("3. Voice Assistant")
                print("4. Job Screening Interview")
                print("5. Startup Pre-screening Interview")  # New option
                print("6. Exit")  # Changed from 5 to 6

                choice = input("Enter choice (1-6): ").strip()

                if choice == "1":
                    self.text_to_speech_mode()
                elif choice == "2":
                    self.text_chat_mode()
                elif choice == "3":
                    self.voice_assistant_mode()
                elif choice == "4":
                    self.job_screening_mode()
                elif choice == "5":
                    self.startup_prescreening_mode()
                elif choice == "6":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice")

                # Reset for next session
                self.convo = self.model.start_chat(history=[])
                self.job_description = ""
                self.screening_questions = []
                self.current_question_index = 0
                self.candidate_responses = []

        except KeyboardInterrupt:
            print("\nSession ended")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.save_transcript()


if __name__ == "__main__":
    assistant = MultilingualAssistant()
    assistant.run()