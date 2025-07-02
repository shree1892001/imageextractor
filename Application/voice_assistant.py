"""
Main application class for the Voice Assistant.
"""
import logging
import re
from typing import Dict, Any, List, Optional

from Core.base import CommandProcessor
from Core.config import Config
from Services.voice_service import VoiceService
from Services.llm_service import GeminiService
from Services.browser_service import PlaywrightBrowserService
from Domain.interaction import WebInteractor


class VoiceWebAssistant(CommandProcessor):
    """Main Voice Web Assistant application class"""
    
    def __init__(self, config: Config):
        """Initialize the Voice Web Assistant"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.voice_service = VoiceService(config)
        self.llm_service = GeminiService(config)
        self.browser_service = PlaywrightBrowserService(config, self.llm_service, self.voice_service)
        
        # Initialize web interactor
        self.web_interactor = WebInteractor(self.browser_service, self.llm_service, self.voice_service)
        
        # Initialize services
        self._initialize_services()
        
        # Set input mode
        self.input_mode = self._get_initial_mode()
        print(f"🚀 Assistant initialized in {self.input_mode} mode")
    
    def _initialize_services(self) -> None:
        """Initialize all services"""
        services = [
            ("Voice Service", self.voice_service),
            ("LLM Service", self.llm_service),
            ("Browser Service", self.browser_service)
        ]
        
        for name, service in services:
            if not service.initialize():
                self.logger.error(f"Failed to initialize {name}")
                raise RuntimeError(f"Failed to initialize {name}")
    
    def _get_initial_mode(self) -> str:
        """Get the initial input mode"""
        print("\n🔊 Select input mode:")
        print("1. Voice\n2. Text")
        while True:
            choice = input("Choice (1/2): ").strip()
            return 'voice' if choice == '1' else 'text'
    
    def process_command(self, command: str) -> bool:
        """Process a command"""
        if not command:
            return True
        
        command_lower = command.lower()
        if command_lower in ["exit", "quit"]:
            return False
        
        if command_lower == "help":
            self._show_help()
            return True
        
        if re.match(r'^(go to|navigate to|open)\s+', command_lower):
            match = re.match(r'^(go to|navigate to|open)\s+(.*)', command, re.IGNORECASE)
            if match:
                url = match.group(2)
                return self.browser_service.navigate(url)
        
        if command_lower in ["text", "voice"]:
            self.voice_service.set_input_mode(command_lower)
            self.voice_service.speak(f"Switched to {command_lower} mode")
            return True
        
        if self._handle_direct_commands(command):
            return True
        
        action_data = self._get_actions(command)
        return self._execute_actions(action_data)
    
    def _handle_direct_commands(self, command: str) -> bool:
        """Handle common commands directly"""
        # Login command
        login_match = re.search(r'login with email\s+(\S+)\s+and password\s+(\S+)', command, re.IGNORECASE)
        if login_match:
            email, password = login_match.groups()
            return self.web_interactor.handle_login(email, password)
        
        # Search command
        search_match = re.search(r'search(?:\s+for)?\s+(.+)', command, re.IGNORECASE)
        if search_match:
            query = search_match.group(1)
            return self.web_interactor.handle_search(query)
        
        # Menu click command
        menu_click_match = re.search(r'click(?:\s+on)?\s+menu\s+item\s+(.+)', command, re.IGNORECASE)
        if menu_click_match:
            menu_item = menu_click_match.group(1)
            return self.web_interactor.handle_menu_navigation(menu_item)
        
        # Submenu navigation command
        submenu_match = re.search(r'navigate(?:\s+to)?\s+(.+?)(?:\s+under|\s+in)?\s+(.+)', command, re.IGNORECASE)
        if submenu_match:
            target_item, parent_menu = submenu_match.groups()
            return self.web_interactor.handle_menu_navigation(target_item, parent_menu)
        
        # Checkbox command
        checkbox_match = re.search(r'(check|uncheck|toggle)(?:\s+the)?\s+(.+)', command, re.IGNORECASE)
        if checkbox_match:
            action, checkbox_label = checkbox_match.groups()
            return self.web_interactor.check_checkbox(
                checkbox_label, 
                checked=(action.lower() == "check" or (action.lower() == "toggle"))
            ).success
        
        # Dropdown selection command
        dropdown_match = re.search(r'select\s+(.+?)(?:\s+from|\s+in)?\s+(.+?)(?:\s+dropdown)?', command, re.IGNORECASE)
        if dropdown_match:
            option, dropdown_name = dropdown_match.groups()
            return self.web_interactor.select_option(dropdown_name, option).success
        
        # State selection command
        state_match = re.search(r'(?:select|choose|pick)\s+(?:state\s+)?(.+)', command, re.IGNORECASE)
        if state_match:
            state_name = state_match.group(1).strip()
            return self.web_interactor.select_state(state_name).success
        
        return False
    
    def _get_actions(self, command: str) -> Dict[str, Any]:
        """Get actions from LLM for a command"""
        context = self.browser_service.get_page_context()
        prompt = self._create_prompt(command, context)
        
        try:
            response = self.llm_service.generate_content(prompt)
            return self._parse_response(response)
        except Exception as e:
            self.logger.error(f"LLM Error: {e}")
            return {"error": str(e)}
    
    def _create_prompt(self, command: str, context: Dict[str, Any]) -> str:
        """Create a prompt for the LLM"""
        input_fields_info = ""
        if "input_fields" in context and context["input_fields"]:
            input_fields_info = "Input Fields Found:\n"
            for idx, field in enumerate(context["input_fields"]):
                input_fields_info += f"{idx + 1}. {field.get('tag', 'input')} - "
                input_fields_info += f"type: {field.get('type', '')}, "
                input_fields_info += f"id: {field.get('id', '')}, "
                input_fields_info += f"name: {field.get('name', '')}, "
                input_fields_info += f"placeholder: {field.get('placeholder', '')}, "
                input_fields_info += f"aria-label: {field.get('aria-label', '')}\n"
        
        menu_items_info = ""
        if "menu_items" in context and context["menu_items"]:
            menu_items_info = "Menu Items Found:\n"
            for idx, item in enumerate(context["menu_items"]):
                submenu_indicator = " (has submenu)" if item.get("has_submenu") else ""
                menu_items_info += f"{idx + 1}. {item.get('text', '')}{submenu_indicator}\n"
        
        buttons_info = ""
        if "buttons" in context and context["buttons"]:
            buttons_info = "Buttons Found:\n"
            for idx, button in enumerate(context["buttons"]):
                buttons_info += f"{idx + 1}. {button.get('text', '')} - "
                buttons_info += f"id: {button.get('id', '')}, "
                buttons_info += f"class: {button.get('class', '')}, "
                buttons_info += f"type: {button.get('type', '')}\n"
        
        return f"""Analyze the web page and generate precise Playwright selectors to complete: \"{command}\".

Selector Priority:
1. ID (
2. Type and Name (input[type='email'], input[name='email'])
3. ARIA labels ([aria-label='Search'])
4. Data-testid ([data-testid='login-btn'])
5. Button text (button:has-text('Sign In'))
6. Semantic CSS classes (.login-button, .p-menuitem)
7. Input placeholder (input[placeholder='Email'])

For tiered menus:
- Parent menus: .p-menuitem, [role='menuitem']
- Submenu items: .p-submenu-list .p-menuitem, ul[role='menu'] [role='menuitem']
- For dropdown/select interactions: Use 'select_option' action when appropriate

Current Page:
Title: {context.get('title', 'N/A')}
URL: {context.get('url', 'N/A')}
Visible Text: {context.get('text', '')[:500]}

{input_fields_info}
{menu_items_info}
{buttons_info}

Relevant HTML:
{context.get('html', '')}

Respond ONLY with JSON in this format:
{{
  "actions": [
    {{
      "action": "click|type|navigate|hover|select_option|check|uncheck|toggle",
      "selector": "CSS selector",
      "text": "(only for type)",
      "purpose": "description",
      "url": "(only for navigate actions)",
      "option": "(only for select_option)",
      "fallback_selectors": ["alternate selector 1", "alternate selector 2"]
    }}
  ]
}}"""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response"""
        try:
            result = re.search(r'\{.*\}', response, re.DOTALL)
            if not result:
                return {"error": "No JSON found in response"}
            
            json_str = result.group(0)
            parsed = self.llm_service.get_structured_guidance(json_str)
            
            if "actions" not in parsed:
                return {"error": "No actions found in response"}
            
            return self._validate_actions(parsed)
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {"error": str(e)}
    
    def _validate_actions(self, actions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the actions from the LLM"""
        valid_actions = []
        
        for action in actions_data.get("actions", []):
            if not self._is_valid_action(action):
                continue
            
            valid_actions.append({
                'action': action['action'].lower(),
                'selector': action.get('selector', ''),
                'text': action.get('text', ''),
                'purpose': action.get('purpose', ''),
                'url': action.get('url', ''),
                'option': action.get('option', ''),
                'fallback_selectors': action.get('fallback_selectors', [])
            })
        
        return {"actions": valid_actions} if valid_actions else {"error": "No valid actions found"}
    
    def _is_valid_action(self, action: Dict[str, Any]) -> bool:
        """Check if an action is valid"""
        requirements = {
            'click': ['selector'],
            'type': ['selector', 'text'],
            'navigate': [],
            'hover': ['selector'],
            'select_option': ['selector', 'option'],
            'check': ['selector'],
            'uncheck': ['selector'],
            'toggle': ['selector']
        }
        
        action_type = action.get('action', '').lower()
        
        if action_type == 'navigate':
            return True
        
        return all(k in action and action[k] is not None for k in requirements.get(action_type, []))
    
    def _execute_actions(self, action_data: Dict[str, Any]) -> bool:
        """Execute actions"""
        if 'error' in action_data:
            self.voice_service.speak("⚠️ Action could not be completed. Switching to fallback...")
            return False
        
        for action in action_data.get('actions', []):
            try:
                self._perform_action(action)
                self.browser_service.page.wait_for_timeout(1000)
            except Exception as e:
                self.voice_service.speak(f"❌ Failed to {action.get('purpose', 'complete action')}")
                self.logger.error(f"Action Error: {str(e)}")
                return False
        
        return True
    
    def _perform_action(self, action: Dict[str, Any]) -> None:
        """Perform an action"""
        action_type = action['action']
        
        if action_type == 'click':
            selector = action.get('selector', '')
            fallbacks = action.get('fallback_selectors', [])
            selectors = [selector] + fallbacks if selector else fallbacks
            self.web_interactor.click(action['purpose'], selectors)
        
        elif action_type == 'type':
            selector = action.get('selector', '')
            fallbacks = action.get('fallback_selectors', [])
            selectors = [selector] + fallbacks if selector else fallbacks
            self.web_interactor.type_text(action['purpose'], action['text'], selectors)
        
        elif action_type == 'navigate':
            url = action.get('url', '')
            if url:
                self.browser_service.navigate(url)
            else:
                purpose = action.get('purpose', '')
                self.web_interactor.click(f"Navigate to {purpose}")
        
        elif action_type == 'hover':
            selector = action.get('selector', '')
            fallbacks = action.get('fallback_selectors', [])
            selectors = [selector] + fallbacks if selector else fallbacks
            self.web_interactor.hover(action['purpose'], selectors)
        
        elif action_type == 'select_option':
            selector = action.get('selector', '')
            option = action.get('option', '')
            fallbacks = action.get('fallback_selectors', [])
            selectors = [selector] + fallbacks if selector else fallbacks
            self.web_interactor.select_option(action['purpose'], option, selectors)
        
        elif action_type in ['check', 'uncheck', 'toggle']:
            selector = action.get('selector', '')
            fallbacks = action.get('fallback_selectors', [])
            selectors = [selector] + fallbacks if selector else fallbacks
            checked = action_type == 'check' or (action_type == 'toggle')
            self.web_interactor.check_checkbox(action['purpose'], checked, selectors)
    
    def _show_help(self) -> None:
        """Show help information"""
        help_text = """
    🔍 Voice Web Assistant Help:

    Basic Navigation:
    - "Go to [website]" - Navigate to a website
    - "Navigate to [section]" - Go to a specific section on the current site
    - "Click on [element]" - Click on a button, link, or other element
    - "Search for [query]" - Use the search function

    Forms:
    - "Type [text] in [field]" - Enter text in an input field
    - "Login with email [email] and password [password]" - Fill login forms
    - "Select [option] from [dropdown]" - Select from dropdown menus
    - "Check/uncheck [checkbox]" - Toggle checkboxes

    Menu Navigation:
    - "Click on menu item [name]" - Click on a menu item
    - "Navigate to [submenu] under [menu]" - Access submenu items

    Input Mode:
    - "Voice" - Switch to voice input mode
    - "Text" - Switch to text input mode

    General:
    - "Help" - Show this help message
    - "Exit" or "Quit" - Close the assistant
    """
        self.voice_service.speak("📋 Showing help")
        print(help_text)
        # Only speak the first part to avoid too much speech
        self.voice_service.engine.say("Here's the help information. You can see the full list on screen.")
        self.voice_service.engine.runAndWait()
    
    def run(self) -> None:
        """Run the assistant"""
        self.voice_service.speak("Web Assistant ready. Say 'help' for available commands.")
        
        # Navigate to default URL
        default_url = self.config.get("default_url", "https://www.google.com")
        self.browser_service.navigate(default_url)
        
        while True:
            command = self.voice_service.listen()
            if not command:
                self.voice_service.speak("I didn't catch that. Please try again.")
                continue
            
            print(f"USER: {command}")
            
            if not self.process_command(command):
                if command.lower() in ["exit", "quit"]:
                    self.voice_service.speak("Goodbye!")
                else:
                    self.voice_service.speak("Something went wrong. Please try again.")
                
                if command.lower() in ["exit", "quit"]:
                    break
    
    def close(self) -> None:
        """Close the assistant and release resources"""
        try:
            self.browser_service.shutdown()
            self.llm_service.shutdown()
            self.voice_service.shutdown()
            print("🛑 Assistant closed")
        except Exception as e:
            self.logger.error(f"Error closing assistant: {e}")
            print(f"Error closing assistant: {e}")
