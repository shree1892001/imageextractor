"""
Base classes and interfaces for the Voice Assistant application.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Service(ABC):
    """Base class for all services"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the service"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the service and release resources"""
        pass


class SpeechService(Service):
    """Interface for speech-related services"""
    
    @abstractmethod
    def speak(self, text: str) -> bool:
        """Convert text to speech"""
        pass
    
    @abstractmethod
    def listen(self) -> str:
        """Listen for speech and convert to text"""
        pass


class BrowserService(Service):
    """Interface for browser automation services"""
    
    @abstractmethod
    def navigate(self, url: str) -> bool:
        """Navigate to a URL"""
        pass
    
    @abstractmethod
    def get_page_context(self) -> Dict[str, Any]:
        """Get current page context"""
        pass
    
    @abstractmethod
    def interact(self, interaction_context: Any) -> Any:
        """Interact with the page"""
        pass


class LLMService(Service):
    """Interface for LLM services"""
    
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generate content using the LLM"""
        pass
    
    @abstractmethod
    def get_structured_guidance(self, prompt: str) -> Dict[str, Any]:
        """Get structured guidance from the LLM"""
        pass


class CommandProcessor(ABC):
    """Interface for command processors"""
    
    @abstractmethod
    def process_command(self, command: str) -> bool:
        """Process a command"""
        pass
