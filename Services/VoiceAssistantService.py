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
import traceback
from TTS.api import TTS
# Suppress warnings
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0


class MultilingualAssistant:
    def __init__(self, api_key=None):
        """Initialize the assistant with improved voice selection"""
        self.api_key = api_key or API_KEY
        
        # Initialize TTS engine first, before any other operations
        self.engine = None
        self.init_tts_engine()
        
        # Initialize other components after voice is set up
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
        self.candidate_name = "Candidate"  # Default name

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize AI model
        self.init_ai_model()

        self.listening = True
        self.active_mode = False
        
        # Test the voice after full initialization
        if self.engine:
            self.speak("Voice assistant initialized with selected voice profile.", wait=True)

    def init_tts_engine(self):
        """Initialize text-to-speech engine with Indian voice if available"""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            
            # Debug: Print all available voices
            print("\nAvailable voices:")
            for i, voice in enumerate(voices):
                print(f"{i}: ID={voice.id}, Name={voice.name}, Languages={voice.languages}")
            
            # Try to find an Indian English voice
            indian_voice_found = False
            
            # First priority: Look for voices with "Indian" or "Hindi" in the name or ID
            for voice in voices:
                if ("indian" in voice.name.lower() or 
                    "hindi" in voice.name.lower() or 
                    "en-in" in voice.id.lower()):
                    self.engine.setProperty('voice', voice.id)
                    print(f"\n✓ Selected Indian voice: {voice.name} (ID: {voice.id})")
                    indian_voice_found = True
                    break
            
            # Second priority: Look for female voices
            if not indian_voice_found:
                for voice in voices:
                    if voice.gender == 'female' or "female" in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        print(f"\n✓ Selected female voice: {voice.name} (ID: {voice.id})")
                        indian_voice_found = True
                        break
            
            # Third priority: Just use the second voice if available (often female)
            if not indian_voice_found and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
                print(f"\n✓ Selected alternate voice: {voices[1].name} (ID: {voices[1].id})")
            
            # Set speech rate and volume
            self.engine.setProperty('rate', 150)  # Slightly slower for clearer pronunciation
            self.engine.setProperty('volume', 0.9)
            
            # Test the selected voice
            print("\nTesting selected voice...")
            self.engine.say("Hello, this is a test of the selected voice.")
            self.engine.runAndWait()
            
            print("Text-to-speech engine initialized successfully")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
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
        self.voice_transcript.append(f"[{timestamp}] Assistant: {cleaned_text}\n    ")

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
        """Save the complete conversation transcript with model answers"""
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
                        f.write(f"   Evaluation Criteria: {q['evaluation']}\n")
                        f.write(f"   Model Answer: {q['model_answer']}\n\n")

                    f.write("\n=== INTERVIEW RESPONSES ===\n")
                    for response in self.candidate_responses:
                        f.write(f"Question: {response['question']}\n")
                        f.write(f"Model Answer: {response['model_answer']}\n")
                        f.write(f"Candidate Response: {response['response']}\n")
                        f.write(f"Evaluation: {response['evaluation']}\n\n")

                    f.write("\n=== FINAL FEEDBACK ===\n")
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
        """Specialized mode for startup pre-screening interviews with personalized greeting"""
        self.speak("Welcome to Startup Pre-screening Mode. Let's start with a brief introduction.")
        
        # Get candidate name
        self.speak("May I know your name, please?")
        candidate_name = self.listen(timeout=15)
        
        if not candidate_name:
            self.speak("I didn't catch your name. Let me call you Candidate for now.")
            candidate_name = "Candidate"
        else:
            # Store the candidate name
            self.candidate_name = candidate_name
            self.speak(f"Thank you, {candidate_name}. It's nice to meet you.")
        
        # Brief introduction
        self.speak(f"I'm an AI assistant conducting this startup pre-screening interview. I'll be asking you questions specifically designed for startup environments.")
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

        # Generate startup-specific questions with model answers
        self.speak("Generating startup-specific screening questions with ideal answers...")
        prompt = f"""Generate 8-10 pre-screening basic questions specifically for a startup environment based on this job description, make sure the questions are basic.
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
        - A model answer that represents an excellent response

        Job Description:
        {self.job_description}

        Return questions in this format:
        CATEGORY: [category]
        QUESTION: [question text]
        IMPORTANCE: [why this matters for startups]
        EVALUATION: [what to look for in answers]
        MODEL_ANSWER: [detailed model answer that represents an excellent response]
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
                    if current_question and "question" in current_question:
                        self.screening_questions.append(current_question)
                    current_question = {
                        "category": line.replace("CATEGORY:", "").strip(),
                        "question": "",
                        "importance": "",
                        "evaluation": "",
                        "model_answer": ""
                    }
                elif line.startswith("QUESTION:"):
                    current_question["question"] = line.replace("QUESTION:", "").strip()
                elif line.startswith("IMPORTANCE:"):
                    current_question["importance"] = line.replace("IMPORTANCE:", "").strip()
                elif line.startswith("EVALUATION:"):
                    current_question["evaluation"] = line.replace("EVALUATION:", "").strip()
                elif line.startswith("MODEL_ANSWER:"):
                    current_question["model_answer"] = line.replace("MODEL_ANSWER:", "").strip()

            if current_question and "question" in current_question:
                self.screening_questions.append(current_question)

            print("\nGenerated Startup Screening Questions:")
            for i, q in enumerate(self.screening_questions, 1):
                print(f"{i}. [{q['category']}] {q['question']}")
                print(f"   Why this matters: {q['importance']}")
                print(f"   Evaluation: {q['evaluation']}")
                print(f"   Model Answer: {q['model_answer']}\n")

            self.speak(f"I've prepared {len(self.screening_questions)} startup-specific questions with model answers.")
            self.speak(f"Alright {candidate_name}, let's begin the interview. I'll be listening for your voice responses.")
            
            # Conduct the interview with voice input
            self.conduct_interview_with_candidate(candidate_name)

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
        """Generate interview questions with model answers from job description"""
        prompt = f"""Based on this job description, generate 7-10 high-quality easy to medium level interview questions that would effectively screen candidates.
        Organize them by category (technical).
        For each question, include:
        1. What a good answer should cover
        2. A detailed model answer that would represent an excellent response

        Job Description:
        {self.job_description}

        Return the questions in this EXACT format:
        CATEGORY: [category name]
        QUESTION: [question text]
        EVALUATION: [what to look for in answers]
        MODEL_ANSWER: [detailed model answer]

        (Repeat the above four lines for each question)
        """

        try:
            response = self.convo.send_message(prompt)
            questions = []

            # Use regex to extract question blocks
            question_blocks = re.findall(
                r"CATEGORY:\s*(.*?)\s*QUESTION:\s*(.*?)\s*EVALUATION:\s*(.*?)\s*MODEL_ANSWER:\s*(.*?)(?=CATEGORY:|$)",
                response.text, re.DOTALL | re.IGNORECASE)

            for category, question, evaluation, model_answer in question_blocks:
                questions.append({
                    "category": category.strip(),
                    "question": question.strip(),
                    "evaluation": evaluation.strip(),
                    "model_answer": model_answer.strip()
                })

            # If regex didn't work, try fallback parsing
            if not questions:
                current_question = {"category": "", "question": "", "evaluation": "", "model_answer": ""}
                for line in response.text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if re.match(r"^(?:CATEGORY:|Category:)\s*(.*)", line, re.IGNORECASE):
                        # If we have a previous question with content, save it
                        if current_question["question"]:
                            questions.append(current_question.copy())

                        # Start a new question
                        current_question = {"category": "", "question": "", "evaluation": "", "model_answer": ""}
                        current_question["category"] = re.match(r"^(?:CATEGORY:|Category:)\s*(.*)", line,
                                                                re.IGNORECASE).group(1).strip()

                    elif re.match(r"^(?:QUESTION:|Question:)\s*(.*)", line, re.IGNORECASE):
                        current_question["question"] = re.match(r"^(?:QUESTION:|Question:)\s*(.*)", line,
                                                                re.IGNORECASE).group(1).strip()

                    elif re.match(r"^(?:EVALUATION:|Evaluation Criteria:)\s*(.*)", line, re.IGNORECASE):
                        current_question["evaluation"] = re.match(r"^(?:EVALUATION:|Evaluation Criteria:)\s*(.*)", line,
                                                                  re.IGNORECASE).group(1).strip()

                    elif re.match(r"^(?:MODEL_ANSWER:|Model Answer:)\s*(.*)", line, re.IGNORECASE):
                        current_question["model_answer"] = re.match(r"^(?:MODEL_ANSWER:|Model Answer:)\s*(.*)", line,
                                                                    re.IGNORECASE).group(1).strip()

                # Add the last question if it has content
                if current_question["question"]:
                    questions.append(current_question)

            print(f"Generated {len(questions)} questions with model answers")
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def get_voice_response(self, question):
        """Get candidate response with enhanced voice input and longer listening time"""
        max_attempts = 3  # Increased from 2
        
        for attempt in range(max_attempts):
            self.speak("Please speak your answer now. I'll listen for up to 60 seconds.")
            print("\nListening for your voice response (up to 60 seconds)...")
            
            try:
                # Create a fresh recognizer for each attempt
                temp_recognizer = sr.Recognizer()
                
                # Use a fresh microphone instance
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise...")
                    # Longer ambient noise adjustment
                    temp_recognizer.adjust_for_ambient_noise(source, duration=2)
                    
                    # More sensitive settings
                    temp_recognizer.energy_threshold = 2500  # Even lower threshold to pick up quieter speech
                    temp_recognizer.dynamic_energy_threshold = True
                    temp_recognizer.dynamic_energy_adjustment_damping = 0.15
                    temp_recognizer.dynamic_energy_adjustment_ratio = 1.5
                    temp_recognizer.pause_threshold = 1.0  # Longer pause threshold
                    
                    print("Listening... (speak clearly and at a normal volume)")
                    # Much longer timeout and phrase time limit
                    audio = temp_recognizer.listen(
                        source, 
                        timeout=60,  # Increased from 20 to 60 seconds
                        phrase_time_limit=120  # Increased to 2 minutes
                    )
                    
                    print("Processing speech...")
                    response = temp_recognizer.recognize_google(
                        audio,
                        language="en-US",  # Explicitly set language
                        show_all=False  # Only return most likely result
                    )
                    
                    if response:
                        print(f"Voice response detected: {response}")
                        
                        # Log the successful voice response
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.voice_transcript.append(f"[{timestamp}] User (voice): {response}\n")
                        self.save_voice_transcript()
                        
                        return response
            
            except sr.WaitTimeoutError:
                print("No speech detected within the 60-second timeout period")
            except sr.UnknownValueError:
                print("Could not understand audio - speech may be too quiet or unclear")
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                print("Falling back to text input due to service error")
                break  # Break immediately on service errors
            except Exception as e:
                print(f"Voice input error: {str(e)}")
                print("Detailed error information:", traceback.format_exc())
            
            if attempt < max_attempts - 1:
                self.speak("I didn't catch that. Let's try again with voice input.")
            else:
                self.speak("Voice input not working. Let's switch to text input.")
        
        # Final fallback to text input
        print("\n" + "="*50)
        print("VOICE INPUT FAILED - SWITCHING TO TEXT INPUT")
        print("="*50)
        self.speak("Please type your answer instead:")
        try:
            text_response = input("Your answer: ").strip()
            if text_response:
                print(f"Text response received: {text_response}")
                
                # Log the text fallback response
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.voice_transcript.append(f"[{timestamp}] User (text fallback): {text_response}\n")
                self.save_voice_transcript()
                
                return text_response
        except Exception as e:
            print(f"Text input error: {str(e)}")
        
        self.speak("No response received. Moving to the next question.")
        return None

    def evaluate_response(self, question, response):
        """Evaluate candidate response against the model answer"""
        if not response or response.strip() == "":
            return "Could not evaluate - no response provided."
        
        prompt = f"""Compare the candidate's answer to the model answer for this question. 
        Focus on semantic meaning rather than exact wording. Give a rating from 1-5 where:
        1: Completely off-topic or incorrect
        2: Some relevant points but missing key elements
        3: Addresses main points but lacks depth
        4: Strong answer covering most key points
        5: Excellent answer matching or exceeding the model answer

        Question: {question['question']}
        Model Answer: {question['model_answer']}
        Candidate Response: {response}

        Return your evaluation in this format:
        Rating: [1-5]
        Matched Concepts: [list key concepts the candidate correctly addressed]
        Missing Elements: [important concepts the candidate didn't mention]
        Unique Insights: [any valuable points the candidate made beyond the model answer]
        Detailed Feedback: [constructive feedback with specific examples]"""

        try:
            # Use your existing conversation/LLM service to evaluate
            evaluation = self.convo.send_message(prompt).text
            return evaluation
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return "Could not evaluate this response due to an error."

    def conduct_interview_with_voice(self):
        """Conduct interview with enhanced voice input"""
        self.speak("Let's begin the interview. I'll ask you questions based on the job description.")
        self.speak(f"There will be {len(self.screening_questions)} questions in total.")
        self.speak("Please answer by speaking clearly. I'll listen for up to 60 seconds for each answer.")
        
        # Add a short pause before starting questions
        time.sleep(2)
        
        for i, question in enumerate(self.screening_questions, 1):
            self.speak(f"\nQuestion {i} of {len(self.screening_questions)}:")
            self.speak(question['question'])
            print(f"\nCategory: {question['category']}")
            print(f"Question: {question['question']}")
            
            # Add a short pause after asking the question
            time.sleep(1)
            
            # Get response with enhanced voice input
            response = self.get_voice_response(question)
            
            if response:
                # Evaluate the response
                evaluation = self.evaluate_response(question, response)
                print("\nEvaluation:")
                print(evaluation)
                
                # Store the response and evaluation
                self.candidate_responses.append({
                    "question": question['question'],
                    "category": question['category'],
                    "response": response,
                    "model_answer": question['model_answer'],
                    "evaluation": evaluation
                })
                
                # Add a short pause between questions
                time.sleep(2)
            else:
                self.speak("Let's move on to the next question.")
                time.sleep(1)
        
        self.generate_interview_feedback()

    def generate_interview_feedback(self, candidate_name="Candidate"):
        """Generate comprehensive interview feedback with personalization"""
        if not self.candidate_responses:
            self.speak("No responses were recorded. Unable to generate feedback.")
            return

        prompt = f"""Based on the candidate's responses to the interview questions, provide comprehensive feedback.
        
        Job Description:
        {self.job_description}
        
        Interview Questions and Responses:
        {json.dumps([{
            'question': r['question'],
            'model_answer': r['model_answer'],
            'candidate_response': r['response'],
            'evaluation': r['evaluation']
        } for r in self.candidate_responses], indent=2)}
        
        Provide feedback in these sections:
        1. Overall Assessment: General impression and fit for the role
        2. Strengths: Key areas where the candidate performed well
        3. Areas for Improvement: Specific skills or knowledge gaps
        4. Final Recommendation: Whether to proceed with the candidate
        
        Address the candidate directly by name: {candidate_name}
        """

        try:
            response = self.convo.send_message(prompt)
            self.interview_feedback = response.text
            
            print("\n=== INTERVIEW FEEDBACK ===")
            print(self.interview_feedback)
            
            self.speak(f"Thank you for completing the interview, {candidate_name}. Here's my feedback:")
            
            # Split feedback into manageable chunks for speaking
            feedback_chunks = re.split(r'(?<=\. )', self.interview_feedback)
            for chunk in feedback_chunks:
                if len(chunk.strip()) > 0:
                    self.speak(chunk)
                    time.sleep(0.5)  # Short pause between sentences
            
            self.speak(f"Thank you for your time, {candidate_name}. The interview is now complete.")
            
            # Save the transcript
            self.save_transcript()
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            self.speak(f"Thank you for completing the interview, {candidate_name}. I encountered an error generating detailed feedback.")

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
        """Enhanced job screening mode with personalized greeting"""
        self.speak("Welcome to the job screening interview. Let's start with a brief introduction.")
        
        # Get candidate name
        self.speak("May I know your name, please?")
        candidate_name = self.listen(timeout=15)
        
        if not candidate_name:
            self.speak("I didn't catch your name. Let me call you Candidate for now.")
            candidate_name = "Candidate"
        else:
            # Store the candidate name
            self.candidate_name = candidate_name
            self.speak(f"Thank you, {candidate_name}. It's nice to meet you.")
        
        # Brief introduction
        self.speak(f"I'm an AI assistant conducting this interview. I'll be asking you questions based on the job description, and evaluating your responses.")
        self.speak("Let's set up the job screening interview. First, I need the job description.")
        
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
        self.conduct_interview_with_candidate(candidate_name)

    def conduct_interview_with_candidate(self, candidate_name):
        """Conduct interview with personalized interaction"""
        self.speak(f"There will be {len(self.screening_questions)} questions in total, {candidate_name}.")
        self.speak("Please answer by speaking clearly. I'll listen for up to 60 seconds for each answer.")
        
        # Add a short pause before starting questions
        time.sleep(2)
        
        for i, question in enumerate(self.screening_questions, 1):
            self.speak(f"\nQuestion {i} of {len(self.screening_questions)}:")
            self.speak(question['question'])
            print(f"\nCategory: {question['category']}")
            print(f"Question: {question['question']}")
            
            # Add a short pause after asking the question
            time.sleep(1)
            
            # Get response with enhanced voice input
            response = self.get_voice_response(question)
            
            if response:
                # Evaluate the response
                evaluation = self.evaluate_response(question, response)
                print("\nEvaluation:")
                print(evaluation)
                
                # Store the response and evaluation
                self.candidate_responses.append({
                    "question": question['question'],
                    "category": question['category'],
                    "response": response,
                    "model_answer": question['model_answer'],
                    "evaluation": evaluation
                })
                
                # Extract rating if possible
                rating_match = re.search(r"Rating:\s*(\d+)", evaluation)
                rating = int(rating_match.group(1)) if rating_match else 0
                
                # Provide personalized feedback based on rating
                if rating >= 4:
                    self.speak(f"That's an excellent answer, {candidate_name}!")
                elif rating == 3:
                    self.speak(f"Good answer, {candidate_name}. You covered the main points.")
                else:
                    self.speak(f"Thank you for your answer, {candidate_name}.")
                
                # Add a short pause between questions
                time.sleep(2)
            else:
                self.speak(f"Let's move on to the next question, {candidate_name}.")
                time.sleep(1)
        
        # Generate and provide feedback
        self.generate_interview_feedback(candidate_name)

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

    def coqui_tts_speak(self, text, lang="hi", output_path="output.wav"):
        """
        Speak text using Coqui TTS for Indian languages.
        lang: Language code, e.g., 'hi' for Hindi, 'en' for English.
        """
        # Choose model based on language
        if lang == "hi":
            model_name = "tts_models/hi/cv/vits"
        elif lang == "en":
            model_name = "tts_models/en/ek1/tacotron2"
        else:
            raise ValueError("Unsupported language for Coqui TTS")

        tts = TTS(model_name)
        tts.tts_to_file(text=text, file_path=output_path)
        # Play the audio (cross-platform)
        import platform
        import os
        if platform.system() == "Windows":
            os.system(f'start {output_path}')
        else:
            os.system(f'play {output_path}')  # Requires sox on Linux/Mac


if __name__ == "__main__":
    assistant = MultilingualAssistant()
    assistant.run()
