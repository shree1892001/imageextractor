import os
import PyPDF2
import pickle
import pyttsx3
from google.generativeai import configure, GenerativeModel
from datetime import datetime
import speech_recognition as sr
from Common.constants import *


class VoiceKnowledgeAssistant:
    def __init__(self, api_key: str):
        # Configure AI model
        configure(api_key=api_key)
        self.model = GenerativeModel("gemini-1.5-flash")

        # Knowledge and memory
        self.knowledge_base = []
        self.memory_file = "assistant_memory.pkl"
        self.conversation_history = []

        # Speech systems
        self.setup_voice_engine()
        self.setup_speech_recognition()

        # Load knowledge base immediately at startup
        self.load_memory()
        print("🌟 Voice Knowledge Assistant Initialized!")

    def setup_voice_engine(self):
        """Initialize text-to-speech with custom settings"""
        try:
            self.tts = pyttsx3.init()
            voices = self.tts.getProperty('voices')
            self.tts.setProperty('rate', 160)
            self.tts.setProperty('volume', 0.9)
            if len(voices) > 1:
                self.tts.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"Error initializing TTS: {e}")

    def setup_speech_recognition(self):
        """Initialize microphone listening"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"Error initializing speech recognition: {e}")

    def load_memory(self):
        """Load saved knowledge and conversation history"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "rb") as f:
                    data = pickle.load(f)
                    self.knowledge_base = data.get("knowledge", [])
                    self.conversation_history = data.get("conversation", [])
                print(f"Loaded {len(self.knowledge_base)} knowledge items")
                if self.knowledge_base:
                    sources = [item["source"] for item in self.knowledge_base]
                    print(f"Knowledge sources: {', '.join(sources)}")
            else:
                print("No memory file found - starting fresh")
        except Exception as e:
            print(f"Error loading memory: {e}")

    def save_memory(self):
        """Save current knowledge and conversation history"""
        try:
            with open(self.memory_file, "wb") as f:
                pickle.dump({
                    "knowledge": self.knowledge_base,
                    "conversation": self.conversation_history
                }, f)
            print("Memory saved successfully")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def speak(self, text: str):
        """Convert text to speech with visual feedback"""
        try:
            print(f"🤖: {text}")
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"Error in speech output: {e}")

    def listen(self) -> str:
        """Capture spoken input with error handling"""
        try:
            with self.microphone as source:
                print("\n[Listening...]")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio)
                print(f"You: {text}")
                return text.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except Exception as e:
            print(f"Recognition error: {e}")
            return ""

    def train_from_file(self, file_path: str):
        """Train from either PDF or text file"""
        try:
            if not os.path.exists(file_path):
                self.speak("File not found")
                return False

            if file_path.lower().endswith('.pdf'):
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages])
            else:  # Assume text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

            # Check if file already exists in knowledge base
            for item in self.knowledge_base:
                if item["source"] == os.path.basename(file_path):
                    self.speak(f"Updating existing knowledge from {os.path.basename(file_path)}")
                    item["content"] = text
                    item["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.save_memory()
                    return True

            # Add new knowledge
            self.knowledge_base.append({
                "source": os.path.basename(file_path),
                "content": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.save_memory()  # Save immediately after adding new knowledge
            self.speak(f"Learned from {os.path.basename(file_path)}")
            return True
        except Exception as e:
            self.speak(f"Error reading file: {str(e)}")
            return False

    def list_knowledge(self):
        """List all available knowledge sources"""
        if not self.knowledge_base:
            self.speak("No knowledge sources available")
            return

        self.speak(f"I have {len(self.knowledge_base)} knowledge sources:")
        for i, item in enumerate(self.knowledge_base, 1):
            self.speak(f"{i}. {item['source']} from {item['timestamp']}")

    def generate_answer(self, question: str):
        """Generate answer based on knowledge base"""
        if not self.knowledge_base:
            self.speak("I don't have any knowledge to answer from. Please train me first.")
            return

        # Add question to conversation history
        self.conversation_history.append({"role": "user", "message": question})

        try:
            # Prepare context from knowledge base
            context = ""
            for item in self.knowledge_base:
                context += f"--- From {item['source']} ---\n"
                # Limit content to reduce token usage if needed
                context += item['content'][:10000] + "\n\n"

            # Create a prompt that utilizes the knowledge base
            prompt = f"""
            Based on the following knowledge base, please answer the question.

            KNOWLEDGE BASE:
            {context}

            QUESTION: {question}

            Provide a clear and concise answer based only on the information in the knowledge base.
            If the answer cannot be found in the knowledge base, state that clearly.
            """

            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            # Add answer to conversation history
            self.conversation_history.append({"role": "assistant", "message": answer})

            # Save updated conversation history
            self.save_memory()

            # Speak the answer
            self.speak(answer)

        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(error_msg)
            self.speak("I'm having trouble answering that question.")

    def process_command(self, command: str):
        """Process user commands"""
        if "train from file" in command:
            self.speak("Please enter the file path:")
            file_path = input("Enter file path: ").strip()
            self.train_from_file(file_path)
        elif "list knowledge" in command:
            self.list_knowledge()
        elif "clear knowledge" in command:
            self.knowledge_base = []
            self.save_memory()
            self.speak("Knowledge base cleared")
        else:
            # Treat as a question to answer
            self.generate_answer(command)


def initialize_knowledge_base():
    """Function to initially set up the knowledge base before starting the assistant"""
    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE SETUP")
    print("=" * 50)

    assistant = VoiceKnowledgeAssistant(api_key=API_KEY_5)

    if not assistant.knowledge_base:
        print("\nNo existing knowledge base found. Let's set one up.")
    else:
        print("\nExisting knowledge base found with the following sources:")
        for i, item in enumerate(assistant.knowledge_base, 1):
            print(f"{i}. {item['source']} (added on {item['timestamp']})")

        choice = input("\nDo you want to add more knowledge? (y/n): ").lower()
        if choice != 'y':
            return assistant

    while True:
        print("\nOptions:")
        print("1. Add a file to knowledge base")
        print("2. List current knowledge")
        print("3. Clear knowledge base")
        print("4. Continue to voice assistant")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            file_path = input("\nEnter file path: ").strip()
            success = assistant.train_from_file(file_path)
            if success:
                print(f"Successfully added {os.path.basename(file_path)} to knowledge base")
        elif choice == '2':
            if not assistant.knowledge_base:
                print("Knowledge base is empty")
            else:
                print("\nCurrent knowledge sources:")
                for i, item in enumerate(assistant.knowledge_base, 1):
                    print(f"{i}. {item['source']} (added on {item['timestamp']})")
        elif choice == '3':
            confirm = input("Are you sure you want to clear the knowledge base? (y/n): ").lower()
            if confirm == 'y':
                assistant.knowledge_base = []
                assistant.save_memory()
                print("Knowledge base cleared")
        elif choice == '4':
            if not assistant.knowledge_base:
                print("\nWARNING: No knowledge added. The assistant won't be able to answer questions.")
                confirm = input("Are you sure you want to continue? (y/n): ").lower()
                if confirm != 'y':
                    continue
            break
        else:
            print("Invalid choice. Please try again.")

    return assistant


if __name__ == "__main__":
    try:
        # First, initialize the knowledge base
        print("\nWelcome to Voice Knowledge Assistant!")
        print("Let's set up your knowledge base before starting.")

        assistant = initialize_knowledge_base()

        # Now start the main voice assistant loop
        print("\n" + "=" * 50)
        print("VOICE ASSISTANT STARTED")
        print("=" * 50)

        assistant.speak("Assistant ready. I can answer questions based on my knowledge base.")
        print("\nCommands:")
        print("- Say 'train from file' to add knowledge (you'll be prompted for the file path)")
        print("- Say 'list knowledge' to see available sources")
        print("- Say 'exit' or 'quit' to end the session")
        print("- Ask any question to get answers from the knowledge base")

        while True:
            user_input = assistant.listen()
            if not user_input:
                continue

            if "exit" in user_input or "quit" in user_input:
                assistant.speak("Goodbye!")
                break

            assistant.process_command(user_input)

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        # Ensure memory is saved before exiting
        if 'assistant' in locals():
            assistant.save_memory()