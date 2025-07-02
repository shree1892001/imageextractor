"""
Configuration management for the Voice Assistant application.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application"""
    
    def __init__(self):
        """Initialize the configuration"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Default configuration
        self._config = {
            # API Keys
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            
            # Voice settings
            "voice_rate": 150,
            "voice_volume": 0.9,
            "voice_id": 1,  # Default to female voice (index 1)
            
            # Browser settings
            "browser_headless": False,
            "browser_slow_mo": 500,
            "viewport_width": 1280,
            "viewport_height": 800,
            
            # LLM settings
            "llm_model": "gemini-1.5-flash",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
            
            # Interaction settings
            "max_retries": 3,
            "retry_delay": 1000,  # milliseconds
            
            # Default startup URL
            "default_url": "https://www.google.com"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.copy()
    
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from a file"""
        try:
            # Implementation depends on file format (JSON, YAML, etc.)
            # For now, just return True
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save_to_file(self, file_path: str) -> bool:
        """Save configuration to a file"""
        try:
            # Implementation depends on file format (JSON, YAML, etc.)
            # For now, just return True
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
