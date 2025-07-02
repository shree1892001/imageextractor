import os
import sys
import PyPDF2
import pickle
import pyttsx3
import speech_recognition as sr
from google.generativeai import configure, GenerativeModel
from datetime import datetime
import subprocess
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from Common.constants import *

WINDOWS_VOICE_NAME = 'Microsoft David Desktop'
TEMPORARY_FILE_PATH = os.path.expanduser('~\\Documents\\mcp_temp')

class ContextProvider(ABC):
    @abstractmethod
    def get_context(self, query: str) -> str:
        pass

    @abstractmethod
    def update_context(self, source: str, content: str):
        pass

class Tool(ABC):
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        pass

class MCPModel:
    def __init__(self, api_key: str):
        configure(api_key=api_key)
        self.model = GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

class WindowsKnowledgeBase(ContextProvider):
    def __init__(self):
        self.knowledge_base: List[Dict] = []
        self.memory_file = os.path.join(TEMPORARY_FILE_PATH, 'mcp_knowledge.pkl')
        self._ensure_temp_dir()
        self.load_memory()

    def _ensure_temp_dir(self):
        if not os.path.exists(TEMPORARY_FILE_PATH):
            os.makedirs(TEMPORARY_FILE_PATH)

    def get_context(self, query: str) -> str:
        return "\n\n".join(
            f"=== {item['source']} ===\n{item['content'][:10000]}"
            for item in self.knowledge_base
        )

    def update_context(self, source: str, content: str):
        existing = next((item for item in self.knowledge_base if item["source"] == source), None)
        if existing:
            existing["content"] = content
            existing["timestamp"] = datetime.now().isoformat()
        else:
            self.knowledge_base.append({
                "source": source,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
        self.save_memory()

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "rb") as f:
                    self.knowledge_base = pickle.load(f)
        except Exception as e:
            print(f"Knowledge Load Error: {e}")

    def save_memory(self):
        try:
            with open(self.memory_file, "wb") as f:
                pickle.dump(self.knowledge_base, f)
        except Exception as e:
            print(f"Knowledge Save Error: {e}")

class WindowsSpeechTool(Tool):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = None
        self._init_windows_audio()

    def _init_windows_audio(self):

        self.tts = pyttsx3.init(driverName='sapi5')
        voices = self.tts.getProperty('voices')

        for voice in voices:
            if WINDOWS_VOICE_NAME in voice.name:
                self.tts.setProperty('voice', voice.id)
                break

        self.tts.setProperty('rate', 160)
        self.tts.setProperty('volume', 0.9)

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def execute(self, input_data: dict) -> Any:
        try:
            if input_data['mode'] == 'speak':
                return self._handle_speech_output(input_data['text'])
            elif input_data['mode'] == 'listen':
                return self._handle_speech_input()
            return ""
        except Exception as e:
            self._log_windows_error(e)
            return ""

    def _handle_speech_output(self, text: str):
        try:
            if not text.strip():
                raise ValueError("Empty text input")

            if not self._check_windows_audio_output():
                raise RuntimeError("Windows audio output not configured properly")

            print(f"Speaking: {text}")
            self.tts.say(text)
            self.tts.runAndWait()
            return True
        except Exception as e:
            self._log_windows_error(e)
            return False

    def _handle_speech_input(self):
        try:
            with self.microphone as source:
                print("Listening... (max 15 seconds)")
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=15
                )

                if not audio or len(audio.get_raw_data()) < 1024:
                    raise ValueError("No audio detected")

                return self.recognizer.recognize_google(audio).lower()
        except sr.WaitTimeoutError:
            print("Listening timed out")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except Exception as e:
            self._log_windows_error(e)
            return ""

    def _check_windows_audio_output(self):
        """Check Windows audio service status"""
        try:
            result = subprocess.run(
                ['powershell', 'Get-Service -Name Audiosrv'],
                capture_output=True,
                text=True
            )
            return "Running" in result.stdout
        except Exception:
            return False

    def _log_windows_error(self, error: Exception):
        error_type = type(error).__name__
        error_msg = str(error)

        diagnostic = f"""
        WINDOWS ERROR DIAGNOSTICS:
        Error Type: {error_type}
        Message: {error_msg}

        System Checks:
        - Microphone Access: {self._check_windows_mic_permissions()}
        - Audio Service: {'Running' if self._check_windows_audio_output() else 'Stopped'}
        - Default Playback Device: {self._get_default_playback_device()}
        """
        print(diagnostic)
        self._suggest_windows_fixes(error_type)

    def _check_windows_mic_permissions(self):
        """Check microphone privacy settings"""
        try:
            result = subprocess.run(
                ['powershell', 'Get-WinUserLanguageList'],
                capture_output=True,
                text=True
            )
            return "Granted" in result.stdout
        except Exception:
            return False

    def _get_default_playback_device(self):
        """Get Windows default playback device"""
        try:
            result = subprocess.run(
                ['powershell', '(Get-AudioDevice -Playback).Name'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() or "Unknown"
        except Exception:
            return "Unknown"

    def _suggest_windows_fixes(self, error_type: str):
        solutions = {
            'ValueError': [
                "Check microphone privacy settings",
                "Run audio troubleshooter: msdt.exe -id AudioPlaybackDiagnostic",
                "Update audio drivers from Device Manager"
            ],
            'RuntimeError': [
                "Restart Windows Audio service: net stop Audiosrv && net start Audiosrv",
                "Check default playback device",
                "Test with Windows Voice Recorder app"
            ],
            'RequestError': [
                "Check internet connection",
                "Disable VPN if active",
                "Temporarily disable Windows Firewall"
            ],
            'default': [
                "Run system file checker: sfc /scannow",
                "Update Windows to latest version",
                "Reinstall audio drivers",
                "Check for hardware conflicts in Device Manager"
            ]
        }

        print(f"Windows Solutions for {error_type}:")
        for i, solution in enumerate(solutions.get(error_type, solutions['default']), 1):
            print(f"{i}. {solution}")

class MCPWindowsAssistant:
    def __init__(self, api_key: str):
        self.mcp_model = MCPModel(api_key)
        self.context_provider = WindowsKnowledgeBase()
        self.tools: Dict[str, Tool] = {
            'speech': WindowsSpeechTool()
        }
        self._preflight_check()

    def _preflight_check(self):
        """Windows-specific system checks"""
        checks = {
            'Audio Service': self._check_audio_service(),
            'Microphone Access': self.tools['speech']._check_windows_mic_permissions(),
            'Temp Directory': os.path.exists(TEMPORARY_FILE_PATH),
            'Internet Connection': self._check_internet()
        }

        print("\nWindows Preflight Check:")
        for check, status in checks.items():
            print(f"{check}: {'OK' if status else 'FAIL'}")

        if not all(checks.values()):
            print("\nSystem configuration issues detected!")
            self.tools['speech'].execute({
                'mode': 'speak',
                'text': "Please check system requirements before continuing"
            })

    def _check_audio_service(self):
        try:
            result = subprocess.run(
                ['sc', 'query', 'Audiosrv'],
                capture_output=True,
                text=True
            )
            return "RUNNING" in result.stdout
        except Exception:
            return False

    def _check_internet(self):
        try:
            subprocess.check_call(
                ['ping', '-n', '1', 'google.com'],
                stdout=subprocess.DEVNULL
            )
            return True
        except Exception:
            return False

    def process_query(self, query: str):
        self.context_provider.update_context(
            source="Conversation History",
            content=f"User: {query}"
        )

        prompt = f"""MCP Windows Prompt v1.2
        Context:
        {self.context_provider.get_context(query)}

        Query: {query}

        Response Requirements:
        1. Prioritize Windows-related context
        2. Format technical terms for speech output
        3. Keep responses under 3 sentences
        """

        response = self.mcp_model.generate(prompt)
        self.tools['speech'].execute({'mode': 'speak', 'text': response})
        return response

    def add_windows_knowledge(self, file_path: str):
        try:
            normalized_path = os.path.normpath(file_path)

            if normalized_path.lower().endswith('.pdf'):
                with open(normalized_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    content = "\n".join([page.extract_text() for page in reader.pages])
            else:
                with open(normalized_path, 'r', encoding='utf-8') as file:
                    content = file.read()

            self.context_provider.update_context(
                source=os.path.basename(normalized_path),
                content=content
            )
            return True
        except Exception as e:
            print(f"Windows File Error: {str(e)}")
            return False

    def voice_interface(self):
        """Windows-optimized voice interaction loop"""
        self.tools['speech'].execute({
            'mode': 'speak',
            'text': "Windows MCP Assistant Ready. Say 'train document' to add knowledge or ask a question."
        })

        while True:
            try:
                command = self.tools['speech'].execute({'mode': 'listen'})

                if 'exit' in command or 'quit' in command:
                    self.tools['speech'].execute({
                        'mode': 'speak',
                        'text': "Closing Windows MCP Assistant"
                    })
                    break

                if 'train document' in command:
                    self._handle_file_training()
                    continue

                if command:
                    self.process_query(command)

            except KeyboardInterrupt:
                print("\nExiting gracefully...")
                break

    def _handle_file_training(self):
        self.tools['speech'].execute({
            'mode': 'speak',
            'text': "Please say the full path to the document"
        })

        file_path = self.tools['speech'].execute({'mode': 'listen'})
        if file_path and self.add_windows_knowledge(file_path):
            self.tools['speech'].execute({
                'mode': 'speak',
                'text': f"Added knowledge from {os.path.basename(file_path)}"
            })
        else:
            self.tools['speech'].execute({
                'mode': 'speak',
                'text': "Failed to process document. Please check the path and try again."
            })

if __name__ == "__main__":

    API_KEY = "your-google-api-key-here"

    if not os.name == 'nt':
        print("This version requires Windows 10 or later")
        sys.exit(1)

    try:
        assistant = MCPWindowsAssistant(api_key=API_KEY_5)
        print("Windows MCP Assistant Active - Press Ctrl+C to exit")
        assistant.voice_interface()
    except Exception as e:
        print(f"Fatal Error: {str(e)}")
        subprocess.run([
            'powershell',
            'Show-Notification -Title "MCP Assistant" -Message "Application crashed. See logs for details."'
        ])