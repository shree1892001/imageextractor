import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass
from Common.constants import *
import asyncio
import json
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


@dataclass
class InteractionContext:
    purpose: str
    element_type: str
    action: str
    value: Optional[str] = None
    selectors: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None


class VoiceSpeaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "english" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

    async def speak(self, text: str):
        print(f"🔊 {text}")
        await asyncio.get_event_loop().run_in_executor(None, self._speak_sync, text)

    def _speak_sync(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()


class VoiceListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    async def listen(self) -> Optional[str]:
        try:
            audio = await asyncio.get_event_loop().run_in_executor(None, self._listen_sync)
            if audio:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self.recognizer.recognize_google, audio
                )
                print(f"👂 Heard: {text}")
                return text.lower()
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results")
            return None
        except Exception as e:
            logging.error(f"Listening error: {str(e)}")
            return None

    def _listen_sync(self):
        try:
            with self.microphone as source:
                print("\n[Listening...]")
                return self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except Exception:
            return None


class LLMSelector:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def get_interaction_details(self, command: str, page_content: str) -> Optional[InteractionContext]:
        prompt = f"""
        Analyze this voice command and webpage content to determine the exact interaction details.
        Command: {command}
        Page Content: {page_content}

        Return a JSON object with these exact keys:
        {{
            "purpose": "description of intended action",
            "element_type": "button/input/link/etc",
            "action": "click/type/select/etc",
            "value": "any input value if needed",
            "selectors": ["selector1", "selector2"],
            "validation": {{"type": "success_check", "criteria": "validation_details"}}
        }}
        """

        try:
            response = await self.model.generate_content(prompt)
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            details = json.loads(response_text)
            return InteractionContext(**details)
        except Exception as e:
            logging.error(f"Failed to get interaction details: {str(e)}")
            return None


class WebInteractor:
    def __init__(self, page: Page, llm_selector: LLMSelector, speaker: VoiceSpeaker):
        self.page = page
        self.llm_selector = llm_selector
        self.speaker = speaker

    async def interact(self, command: str) -> bool:
        try:
            page_content = await self.page.content()
            context = await self.llm_selector.get_interaction_details(command, page_content)

            if not context:
                await self.speaker.speak("I couldn't understand how to interact with the page")
                return False

            for selector in (context.selectors or []):
                try:
                    element = await self.page.wait_for_selector(selector, timeout=2000)
                    if not element:
                        continue

                    if context.action == "click":
                        await element.click()
                    elif context.action == "type":
                        await element.fill(context.value or "")
                    elif context.action == "select":
                        await element.select_option(value=context.value)
                    elif context.action == "checkbox":
                        await element.set_checked(context.value.lower() == "true")

                    if context.validation:
                        if not await self._validate_action(context):
                            continue

                    await self.speaker.speak(f"Successfully performed {context.action} action")
                    return True

                except Exception as e:
                    logging.debug(f"Selector {selector} failed: {str(e)}")
                    continue

            await self.speaker.speak("I couldn't find the right element to interact with")
            return False

        except Exception as e:
            logging.error(f"Interaction failed: {str(e)}")
            await self.speaker.speak("Sorry, something went wrong")
            return False

    async def _validate_action(self, context: InteractionContext) -> bool:
        try:
            page_state = await self.page.content()
            prompt = f"""
            Validate if this interaction was successful:
            Action: {context.action}
            Validation Rules: {context.validation}
            Current Page State: {page_state}

            Return exactly "true" or "false"
            """
            response = await self.llm_selector.model.generate_content(prompt)
            return response.text.strip().lower() == "true"
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False


class VoiceWebAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm_selector = LLMSelector(api_key)
        self.speaker = VoiceSpeaker()
        self.listener = VoiceListener()
        self.input_mode = "voice"  # Default mode
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def choose_input_mode(self):
        print("\n🔊 Select input mode:")
        print("1. Voice")
        print("2. Text")
        while True:
            choice = input("Choice (1/2): ").strip()
            if choice == '1':
                self.input_mode = "voice"
                return "voice"
            elif choice == '2':
                self.input_mode = "text"
                return "text"
            else:
                print("Invalid choice. Please select 1 or 2.")

    async def get_command(self) -> Optional[str]:
        command = None

        if self.input_mode == "voice":
            command = await self.listener.listen()
        else:  # text mode
            try:
                command = input("\n⌨️ Command: ").strip().lower()
                if not command:  # Handle empty input
                    return None
            except Exception as e:
                logging.error(f"Text input error: {str(e)}")
                return None

        # Allow switching modes on the fly
        if command and command in ["text mode", "text"]:
            self.input_mode = "text"
            await self.speaker.speak("Switched to text input mode.")
            return None
        elif command and command in ["voice mode", "voice"]:
            self.input_mode = "voice"
            await self.speaker.speak("Switched to voice input mode.")
            return None

        return command

    async def start(self):
        logging.basicConfig(level=logging.INFO)

        try:
            async with async_playwright() as p:
                self.browser = await p.chromium.launch(headless=False)
                self.context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720}
                )
                self.page = await self.context.new_page()

                selected_mode = self.choose_input_mode()
                await self.speaker.speak(f"{selected_mode.capitalize()} mode activated.")

                interactor = WebInteractor(self.page, self.llm_selector, self.speaker)
                await self.speaker.speak("Voice Web Assistant ready. Say 'open' followed by a website to begin.")

                while True:
                    command = await self.get_command()
                    if not command:
                        continue

                    if command in ["quit", "exit", "stop"]:
                        await self.speaker.speak("Goodbye!")
                        break

                    if command.startswith("open"):
                        url = command.replace("open", "").strip()
                        if not url.startswith(("http://", "https://")):
                            url = "https://" + url
                        try:
                            await self.page.goto(url)
                            await self.speaker.speak(f"Opened {url}")
                        except Exception as e:
                            await self.speaker.speak(f"Failed to open {url}")
                            logging.error(f"Navigation failed: {str(e)}")
                        continue

                    await interactor.interact(command)

        except Exception as e:  # Fixed the missing colon here
            logging.error(f"Assistant execution failed: {str(e)}")
            await self.speaker.speak("An error occurred. Shutting down.")
        finally:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()


async def main():
    assistant = VoiceWebAssistant(API_KEY_5)
    await assistant.start()


if __name__ == "__main__":
    asyncio.run(main())