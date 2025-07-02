import pyttsx3
from typing import Optional

class Speaker:
    def __init__(self, voice_id: Optional[str] = None, rate: int = 150):

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        
        # Set voice if specified
        if voice_id:
            self.engine.setProperty('voice', voice_id)
        else:
            # Try to set a default English voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if "english" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

    def speak(self, text: str) -> None:

        try:
            print(f"🔊 {text}")  # Visual feedback
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {str(e)}")
            # Fallback to just printing if speech fails
            print(f"Message: {text}")

    def list_available_voices(self) -> list:
        """
        Get list of available voices
        
        Returns:
            List of available voice IDs and names
        """
        voices = self.engine.getProperty('voices')
        return [(voice.id, voice.name) for voice in voices]

    def set_rate(self, rate: int) -> None:
        """
        Set speech rate
        
        Args:
            rate: Words per minute
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """
        Set speech volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))