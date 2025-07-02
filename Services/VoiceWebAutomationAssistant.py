from playwright.sync_api import sync_playwright
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
from Common.constants import *

@dataclass
class AutomationCommand:
    """Structure for voice commands"""
    action: str
    element_type: str
    value: Optional[str] = None
    purpose: Optional[str] = None

class VoiceWebAutomationAssistant:
    def __init__(self, gemini_api_key: str):
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self._setup_voice_engine()
        
        # Initialize Playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False, slow_mo=500)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 800})
        self.page = self.context.new_page()
        
        self.logger = logging.getLogger(__name__)

    def _setup_voice_engine(self):
        """Initialize text-to-speech engine"""
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

    def speak(self, text: str):
        """Text-to-speech output"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> str:
        """Listen for voice input"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                self.speak("Could not understand audio")
                return ""
            except sr.RequestError:
                self.speak("Could not request results")
                return ""

    def parse_command(self, voice_input: str) -> AutomationCommand:
        """Parse voice input into automation command"""
        prompt = f"""
        Convert this voice command into a structured automation command:
        "{voice_input}"
        
        Extract:
        1. Action (click, type, select, hover, checkbox)
        2. Element type (button, input, dropdown, link)
        3. Value (if any)
        4. Purpose (description of the element)
        
        Return as JSON with these exact keys:
        action, element_type, value, purpose
        """
        
        response = self.llm.generate_content(prompt)
        try:
            command_dict = eval(response.text)
            return AutomationCommand(**command_dict)
        except Exception as e:
            self.logger.error(f"Failed to parse command: {e}")
            return None

    async def execute_command(self, command: AutomationCommand):
        """Execute the automation command"""
        context = InteractionContext(
            purpose=command.purpose,
            element_type=command.element_type,
            action=command.action,
            value=command.value
        )
        
        interactor = WebInteractor(self.page, self.llm, self)
        success = await interactor.interact(context)
        
        if success:
            self.speak(f"Successfully completed {command.action} action")
        else:
            self.speak(f"Failed to complete {command.action} action")

    def run(self):
        """Main loop for voice automation"""
        self.speak("Voice Web Automation Assistant is ready")
        
        while True:
            self.speak("Waiting for your command")
            voice_input = self.listen()
            
            if not voice_input:
                continue
                
            if "exit" in voice_input or "quit" in voice_input:
                self.speak("Goodbye!")
                break
                
            if "open website" in voice_input:
                url = voice_input.split("open website")[-1].strip()
                self.page.goto(url)
                self.speak(f"Opened {url}")
                continue
            
            command = self.parse_command(voice_input)
            if command:
                await self.execute_command(command)
            else:
                self.speak("I couldn't understand that command")

    def cleanup(self):
        """Cleanup resources"""
        self.browser.close()
        self.playwright.stop()

if __name__ == "__main__":
    assistant = VoiceWebAutomationAssistant(API_KEY_5)
    try:
        assistant.run()
    finally:
        assistant.cleanup()