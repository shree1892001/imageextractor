from playwright.sync_api import sync_playwright
import speech_recognition as sr
import pyttsx3
import time
import os
import re


class VoiceWebAssistant:
    def __init__(self):
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.setup_text_to_speech()

        # Initialize Playwright browser
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.setup_browser()

        # Set initial input mode
        self.use_voice_input = True

        print("🎙️ Voice Web Assistant initialized!")

    def setup_text_to_speech(self):
        """Setup the text-to-speech engine"""
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.save_audio = False
        self.audio_dir = "speech_output"
        if self.save_audio and not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)

    def setup_browser(self):
        """Initialize Playwright browser"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        print("🌐 Browser ready")

    def speak(self, text):
        """Convert text to speech"""
        print(f"ASSISTANT: {text}")
        if self.save_audio:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.audio_dir, f"speech_{timestamp}.mp3")
            self.engine.save_to_file(text, filename)
        self.engine.say(text)
        self.engine.runAndWait()

    def read_webpage_content(self):
        """Read the main content of the current webpage"""
        try:
            content = ""
            # Try different content selectors
            selectors = ["main", "article", "#content", ".content", "body"]
            for selector in selectors:
                element = self.page.locator(selector)
                if element.count() > 0:
                    content = element.all_text_contents()
                    break

            if not content:
                content = self.page.locator("body").all_text_contents()

            cleaned_content = ' '.join(content).strip()
            cleaned_content = re.sub(r'\n+', '\n', cleaned_content)
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)

            if len(cleaned_content) > 500:
                self.speak("Here's the beginning of the page content:")
                self.speak(cleaned_content[:500] + "... (content truncated)")
            else:
                self.speak("Here's the page content:")
                self.speak(cleaned_content)
            return True
        except Exception as e:
            self.speak(f"I couldn't read the page content. Error: {str(e)}")
            return False

    def listen_voice(self):
        """Capture voice input from microphone"""
        try:
            with self.microphone as source:
                print("\n[Listening...]")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio)
                print(f"YOU: {text}")
                return text.lower()
        except sr.UnknownValueError:
            self.speak("I didn't catch that. Could you repeat?")
            return ""
        except sr.RequestError:
            self.speak("Speech service unavailable. Switching to text input.")
            self.use_voice_input = False
            return self.listen_text()
        except sr.WaitTimeoutError:
            self.speak("I didn't hear anything. Please speak when ready.")
            return ""
        except Exception as e:
            print(f"Listening error: {e}")
            self.speak("Listening error. Switching to text input.")
            self.use_voice_input = False
            return self.listen_text()

    def listen_text(self):
        """Get input from text"""
        try:
            print("\n[Text Input Mode: Type command or 'voice mode']")
            text_input = input("YOU: ").strip()
            if text_input.lower() == "voice mode":
                self.use_voice_input = True
                self.speak("Switching to voice input mode.")
                return ""
            return text_input.lower()
        except Exception as e:
            print(f"Text input error: {e}")
            return ""

    def listen(self):
        """Get input from current mode"""
        return self.listen_voice() if self.use_voice_input else self.listen_text()

    def browse_website(self, url):
        """Navigate to a website with enhanced URL handling"""
        # Clean URL
        url = re.sub(r'[^a-zA-Z0-9\.\-/:]', '', url)
        replacements = {
            " dot ": ".", " slash ": "/", " colon ": ":",
            " com ": ".com", " org ": ".org", " net ": ".net"
        }
        for word, replacement in replacements.items():
            url = url.replace(word, replacement)

        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}' if url.startswith('www.') else f'https://www.{url}'

        self.speak(f"Opening {url}")
        try:
            self.page.goto(url)
            self.page.wait_for_load_state("networkidle")
            title = self.page.title()
            self.speak(f"Successfully opened: {title}")
            return True
        except Exception as e:
            self.speak(f"Couldn't access website. Error: {str(e)}")
            return False

    def find_element_smart(self, target_text):
        """Find elements with case-insensitive matching"""
        try:
            # Try different strategies
            strategies = [
                lambda: self.page.get_by_text(target_text, exact=True),
                lambda: self.page.get_by_text(target_text),
                lambda: self.page.locator(f"text=/{re.escape(target_text)}/i"),
                lambda: self.page.get_by_label(target_text),
                lambda: self.page.get_by_placeholder(target_text),
                lambda: self.page.get_by_role("button", name=target_text)
            ]

            for strategy in strategies:
                try:
                    element = strategy()
                    if element.count() > 0:
                        element.first.wait_for(state="visible")
                        return element.first
                except:
                    continue
            return None
        except Exception as e:
            self.speak(f"Element not found: {str(e)}")
            return None

    def perform_web_action(self, action, target=None, value=None):
        """Execute web automation commands"""
        try:
            if action == "click":
                if element := self.find_element_smart(target):
                    element.click()
                    self.speak(f"Clicked on {target}")
                    return True
            elif action == "type":
                if element := self.find_element_smart(target):
                    element.fill(value)
                    self.speak(f"Typed: {value}")
                    return True
            elif action == "search":
                search_box = self.page.get_by_role("searchbox") or \
                             self.page.locator("[type='search']") or \
                             self.page.locator("[name='q']") or \
                             self.page.locator("[name='search']")

                if search_box.count() > 0:
                    search_box.first.fill(value)
                    search_box.first.press("Enter")
                    self.speak(f"Searching for {value}")
                    return True
                else:
                    self.speak("Couldn't find search box")
            return False
        except Exception as e:
            self.speak(f"Action failed. Error: {str(e)}")
            return False

    def process_command(self, command):
        """Interpret and execute commands"""
        if not command:
            return

        # Mode switching
        if "text mode" in command:
            self.use_voice_input = False
            self.speak("Text input mode activated.")
            return
        if "voice mode" in command:
            self.use_voice_input = True
            self.speak("Voice input mode activated.")
            return

        # Help command
        if "help" in command:
            self.display_help()
            return

        # Website navigation
        if (site := self.extract_site_from_command(command)):
            self.browse_website(site)
            return

        # Click actions
        if (click_target := self.extract_click_target(command)):
            self.perform_web_action("click", click_target)
            return

        # Form interactions
        form_data = self.extract_form_data(command)
        if form_data and form_data[0] and form_data[1]:
            text, field = form_data
            self.perform_web_action("type", field, text)
            return

        # Web searches
        if (query := self.extract_search_query(command)):
            self.perform_web_action("search", value=query)
            return

        # Read content
        if "read" in command and ("page" in command or "content" in command):
            self.read_webpage_content()
            return

        self.speak("Command not recognized. Say 'help' for options.")

    def extract_site_from_command(self, command):
        """Extract website from command"""
        patterns = [
            r'go to (["\']?)(.+?)\1',
            r'visit (["\']?)(.+?)\1',
            r'open (["\']?)(.+?)\1'
        ]
        for pattern in patterns:
            if match := re.search(pattern, command, re.IGNORECASE):
                return re.sub(r'[^a-zA-Z0-9\.\-]', '', match.group(2))
        return None

    def extract_click_target(self, command):
        """Extract click target from command"""
        if "click on " in command:
            return command.split("click on ")[1].strip()
        if "click " in command:
            return command.split("click ")[1].strip()
        return None

    def extract_form_data(self, command):
        """Extract form data from command"""
        if "type " in command and " in " in command:
            parts = command.split("type ")[1].split(" in ")
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
        return None, None

    def extract_search_query(self, command):
        """Extract search query from command"""
        if "search for " in command:
            return command.split("search for ")[1].strip()
        if "search " in command:
            return command.split("search ")[1].strip()
        return None

    def display_help(self):
        """Show available commands"""
        help_text = """
Available commands:
- Navigation: "go to [website]", "visit [url]"
- Interaction: "click [element]", "type [text] in [field]"
- Search: "search for [query]"
- Content: "read page"
- Modes: "text mode", "voice mode"
- System: "help", "exit"
"""
        print(help_text)
        self.speak("Here are the available commands.")

    def run(self):
        """Main interaction loop"""
        self.speak("Hello! I'm your voice web assistant. How can I help you?")
        self.speak("Say 'help' for command list.")

        try:
            while True:
                try:
                    command = self.listen()
                    if command in ["exit", "quit", "stop"]:
                        self.speak("Goodbye!")
                        break
                    self.process_command(command)
                except KeyboardInterrupt:
                    self.speak("Shutting down...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    self.speak("Let's try again.")
        finally:
            self.context.close()
            self.browser.close()
            self.playwright.stop()
            print("Browser closed")


if __name__ == "__main__":
    try:
        print("""
╔════════════════════════════════════════════════════════╗
║                 VOICE WEB ASSISTANT                     ║
║ --------------------------------------------------------║
║ Commands: "go to [website]", "click [element]",         ║
║ "type [text] in [field]", "search [query]", "read page" ║
╚════════════════════════════════════════════════════════╝
""")
        assistant = VoiceWebAssistant()
        assistant.run()
    except Exception as e:
        print(f"Critical error: {e}")
        print("Please check browser drivers and dependencies.")