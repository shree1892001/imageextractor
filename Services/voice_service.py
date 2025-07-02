"""
Voice recognition and text-to-speech service.
"""
import pyttsx3
import speech_recognition as sr
import logging
from typing import Optional

from Core.base import SpeechService
from Core.config import Config


class VoiceService(SpeechService):
    """Service for voice recognition and text-to-speech"""
    
    def __init__(self, config: Config):
        """Initialize the voice service"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self.recognizer = None
        self.microphone = None
        self.input_mode = 'voice'  # Default to voice input
    
    def initialize(self) -> bool:
        """Initialize the voice service"""
        try:
            # Initialize text-to-speech
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            voice_id = self.config.get('voice_id', 1)
            if len(voices) > voice_id:
                self.engine.setProperty('voice', voices[voice_id].id)
            self.engine.setProperty('rate', self.config.get('voice_rate', 150))
            self.engine.setProperty('volume', self.config.get('voice_volume', 0.9))
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            self.logger.info("Voice service initialized")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing voice service: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the voice service"""
        try:
            # No specific cleanup needed for pyttsx3 or speech_recognition
            self.logger.info("Voice service shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Error shutting down voice service: {e}")
            return False
    
    def speak(self, text: str) -> bool:
        """Convert text to speech"""
        try:
            print(f"ASSISTANT: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"Error speaking: {e}")
            return False
    
    def listen(self) -> str:
        """Listen for speech and convert to text"""
        if self.input_mode == 'voice':
            return self._listen_voice()
        return self._listen_text()
    
    def _listen_voice(self) -> str:
        """Listen for voice input"""
        try:
            with self.microphone as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=5, 
                    phrase_time_limit=10
                )
                text = self.recognizer.recognize_google(audio).lower()
                print(f"USER: {text}")
                return text
        except sr.UnknownValueError:
            self.logger.debug("Speech not recognized")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Error listening: {e}")
            return ""
    
    def _listen_text(self) -> str:
        """Get text input from the user"""
        try:
            text = input("\n⌨️ Command: ").strip()
            if text.lower() in ["voice", "voice mode"]:
                self.input_mode = 'voice'
                self.speak("Voice mode activated")
            return text
        except Exception as e:
            self.logger.error(f"Error getting text input: {e}")
            return ""
    
    def set_input_mode(self, mode: str) -> None:
        """Set the input mode (voice or text)"""
        if mode.lower() in ['voice', 'text']:
            self.input_mode = mode.lower()
            self.logger.info(f"Input mode set to {mode}")
        else:
            self.logger.warning(f"Invalid input mode: {mode}")
    
    def get_input_mode(self) -> str:
        """Get the current input mode"""
        return self.input_mode
