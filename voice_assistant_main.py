"""
Main entry point for the Voice Web Assistant application.
"""
import os
import sys
import logging
from dotenv import load_dotenv

from Core.config import Config
from Core.utils import setup_logging
from Application.voice_assistant import VoiceWebAssistant
from Common.constants import API_KEY_3


def main():
    """Main entry point"""
    # Set up logging
    logger = setup_logging()
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = Config()
    
    # Set API key from constants if not in environment
    if not config.get("gemini_api_key"):
        config.set("gemini_api_key", API_KEY_3)
    
    # Check if API key is available
    if not config.get("gemini_api_key"):
        logger.error("No API key provided for Gemini")
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file with your API key or set it in your environment.")
        sys.exit(1)
    
    # Create and run the assistant
    try:
        assistant = VoiceWebAssistant(config)
        assistant.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"Error running assistant: {e}")
        print(f"❌ Error: {e}")
    finally:
        try:
            assistant.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
