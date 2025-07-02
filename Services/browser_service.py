"""
Browser automation service.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext

from Core.base import BrowserService
from Core.config import Config
from Core.utils import filter_html
from Domain.models import InteractionContext, InteractionResult, PageContext


class PlaywrightBrowserService(BrowserService):
    """Service for browser automation using Playwright"""
    
    def __init__(self, config: Config, llm_service: Any, voice_service: Any):
        """Initialize the browser service"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service
        self.voice_service = voice_service
        
        # Playwright components
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Interaction settings
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1000)  # milliseconds
    
    def initialize(self) -> bool:
        """Initialize the browser service"""
        try:
            self.playwright = sync_playwright().start()
            
            # Launch browser
            self.browser = self.playwright.chromium.launch(
                headless=self.config.get("browser_headless", False),
                slow_mo=self.config.get("browser_slow_mo", 500)
            )
            
            # Create browser context
            self.context = self.browser.new_context(
                viewport={
                    'width': self.config.get("viewport_width", 1280),
                    'height': self.config.get("viewport_height", 800)
                }
            )
            
            # Create page
            self.page = self.context.new_page()
            
            self.logger.info("Browser service initialized")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing browser service: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the browser service"""
        try:
            if self.context:
                self.context.close()
            
            if self.browser:
                self.browser.close()
            
            if self.playwright:
                self.playwright.stop()
            
            self.logger.info("Browser service shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Error shutting down browser service: {e}")
            return False
    
    def navigate(self, url: str) -> bool:
        """Navigate to a URL"""
        try:
            # Handle different URL formats
            if "://" in url:
                self.voice_service.speak(f"🌐 Navigating to {url}")
                self.page.goto(url, wait_until="networkidle", timeout=20000)
            elif url.startswith('#') or url.startswith('/#'):
                current_url = self.page.url
                base_url = current_url.split('#')[0]
                new_url = f"{base_url}{url}" if url.startswith('#') else f"{base_url}{url[1:]}"
                self.voice_service.speak(f"🌐 Navigating within page to {url}")
                self.page.goto(new_url, wait_until="networkidle", timeout=20000)
            elif not url.startswith(('http://', 'https://')):
                if "/" in url and not url.startswith("/"):
                    domain = url.split("/")[0]
                    self.voice_service.speak(f"🌐 Navigating to https://{domain}")
                    self.page.goto(f"https://{domain}", wait_until="networkidle", timeout=20000)
                else:
                    self.voice_service.speak(f"🌐 Navigating to https://{url}")
                    self.page.goto(f"https://{url}", wait_until="networkidle", timeout=20000)
            else:
                current_url = self.page.url
                domain_match = re.match(r'^(?:http|https)://[^/]+', current_url)
                if domain_match:
                    domain = domain_match.group(0)
                    new_url = f"{domain}/{url}"
                    self.voice_service.speak(f"🌐 Navigating to {new_url}")
                    self.page.goto(new_url, wait_until="networkidle", timeout=20000)
                else:
                    self.voice_service.speak(f"🌐 Navigating to https://{url}")
                    self.page.goto(f"https://{url}", wait_until="networkidle", timeout=20000)
            
            self.voice_service.speak(f"📄 Loaded: {self.page.title()}")
            self._dismiss_popups()
            return True
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            self.voice_service.speak(f"❌ Navigation failed: {str(e)}")
            return False
    
    def get_page_context(self) -> Dict[str, Any]:
        """Get current page context"""
        try:
            self.page.wait_for_timeout(1000)
            
            # Get input fields
            input_fields = []
            inputs = self.page.locator("input:visible, textarea:visible, select:visible")
            count = inputs.count()
            
            for i in range(min(count, 10)):
                try:
                    field = inputs.nth(i)
                    field_info = {
                        "tag": field.evaluate("el => el.tagName.toLowerCase()"),
                        "type": field.evaluate("el => el.type || ''"),
                        "id": field.evaluate("el => el.id || ''"),
                        "name": field.evaluate("el => el.name || ''"),
                        "placeholder": field.evaluate("el => el.placeholder || ''"),
                        "aria-label": field.evaluate("el => el.getAttribute('aria-label') || ''")
                    }
                    input_fields.append(field_info)
                except Exception:
                    pass
            
            # Get menu items
            menu_items = []
            try:
                menus = self.page.locator(
                    "[role='menubar'] [role='menuitem'], .p-menuitem, nav a, .navigation a, .menu a, header a")
                menu_count = menus.count()
                
                for i in range(min(menu_count, 20)):
                    try:
                        menu_item = menus.nth(i)
                        text = menu_item.inner_text().strip()
                        if text:
                            has_submenu = menu_item.locator(
                                ".p-submenu-icon, [class*='submenu'], [class*='dropdown'], [class*='caret']").count() > 0
                            menu_items.append({
                                "text": text,
                                "has_submenu": has_submenu
                            })
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Get buttons
            buttons = []
            try:
                button_elements = self.page.locator(
                    "button:visible, [role='button']:visible, input[type='submit']:visible, input[type='button']:visible")
                button_count = button_elements.count()
                
                for i in range(min(button_count, 10)):
                    try:
                        button = button_elements.nth(i)
                        text = button.inner_text().strip()
                        buttons.append({
                            "text": text,
                            "id": button.evaluate("el => el.id || ''"),
                            "class": button.evaluate("el => el.className || ''"),
                            "type": button.evaluate("el => el.type || ''")
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Create page context
            context = PageContext(
                title=self.page.title(),
                url=self.page.url,
                text=self.page.locator("body").inner_text()[:1000],
                html=filter_html(self.page.locator("body").inner_html()[:4000]),
                input_fields=input_fields,
                menu_items=menu_items,
                buttons=buttons
            )
            
            return context.__dict__
        except Exception as e:
            self.logger.error(f"Error getting page context: {e}")
            return {
                "title": "Error",
                "url": "Error",
                "text": f"Error getting page context: {str(e)}",
                "html": "",
                "input_fields": [],
                "menu_items": [],
                "buttons": []
            }
    
    def interact(self, context: InteractionContext) -> InteractionResult:
        """Interact with the page"""
        # Map interaction methods
        interaction_methods = {
            "click": self._handle_click,
            "type": self._handle_type,
            "select": self._handle_select,
            "hover": self._handle_hover,
            "checkbox": self._handle_checkbox
        }
        
        # Get the appropriate handler
        handler = interaction_methods.get(context.action)
        if not handler:
            error_msg = f"Unsupported action: {context.action}"
            self.logger.error(error_msg)
            return InteractionResult(success=False, message=error_msg)
        
        # Execute the handler
        try:
            result = handler(context)
            return result
        except Exception as e:
            error_msg = f"Error executing {context.action}: {str(e)}"
            self.logger.error(error_msg)
            return InteractionResult(success=False, message=error_msg)
    
    def _dismiss_popups(self) -> None:
        """Dismiss common popups like cookie notices"""
        try:
            context = self.get_page_context()
            popup_selectors = self.llm_service.get_selectors(
                "find popup close button, cookie acceptance button, or dismiss button", 
                context
            )
            
            for selector in popup_selectors:
                try:
                    if self.page.locator(selector).count() > 0:
                        self.page.locator(selector).first.click(timeout=2000)
                        self.voice_service.speak("🗑️ Closed popup")
                        self.page.wait_for_timeout(1000)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    
    def _handle_click(self, context: InteractionContext) -> InteractionResult:
        """Handle click interaction"""
        guidance = self._get_llm_guidance(context)
        
        for selector in guidance.get("selectors", []):
            if self._retry_action(self._click_element, selector, context.purpose):
                return InteractionResult(
                    success=True,
                    message=f"Clicked {context.purpose}"
                )
        
        return InteractionResult(
            success=False,
            message=f"Failed to click {context.purpose}"
        )
    
    def _handle_type(self, context: InteractionContext) -> InteractionResult:
        """Handle type interaction"""
        guidance = self._get_llm_guidance(context)
        
        for selector in guidance.get("selectors", []):
            if self._retry_action(self._type_text, selector, context.value, context.purpose):
                return InteractionResult(
                    success=True,
                    message=f"Typed text in {context.purpose}"
                )
        
        return InteractionResult(
            success=False,
            message=f"Failed to type text in {context.purpose}"
        )
    
    def _handle_select(self, context: InteractionContext) -> InteractionResult:
        """Handle select interaction"""
        guidance = self._get_llm_guidance(context)
        
        # Handle special handling steps if provided
        if "special_handling" in guidance:
            for step in guidance["special_handling"]:
                self._execute_step(step)
        
        # Try selectors
        for selector in guidance.get("selectors", []):
            if self._retry_action(self._select_option, selector, context.value, context.purpose):
                # Verify selection if verification steps provided
                if "verification" in guidance:
                    if self._verify_selection(guidance["verification"], context):
                        return InteractionResult(
                            success=True,
                            message=f"Selected {context.value} from {context.purpose}"
                        )
                else:
                    return InteractionResult(
                        success=True,
                        message=f"Selected {context.value} from {context.purpose}"
                    )
        
        return InteractionResult(
            success=False,
            message=f"Failed to select {context.value} from {context.purpose}"
        )
    
    def _handle_hover(self, context: InteractionContext) -> InteractionResult:
        """Handle hover interaction"""
        guidance = self._get_llm_guidance(context)
        
        for selector in guidance.get("selectors", []):
            if self._retry_action(self._hover_element, selector, context.purpose):
                return InteractionResult(
                    success=True,
                    message=f"Hovered over {context.purpose}"
                )
        
        return InteractionResult(
            success=False,
            message=f"Failed to hover over {context.purpose}"
        )
    
    def _handle_checkbox(self, context: InteractionContext) -> InteractionResult:
        """Handle checkbox interaction"""
        guidance = self._get_llm_guidance(context)
        
        if context.product_name:
            # Handle product selection specifically
            result = self._select_product(context.product_name, context.value == 'true')
            return InteractionResult(
                success=result,
                message=f"{'Selected' if result else 'Failed to select'} {context.product_name}"
            )
        
        # Default checkbox handling
        for selector in guidance.get("selectors", []):
            if self._retry_action(self._toggle_checkbox, selector, context.value, context.purpose):
                return InteractionResult(
                    success=True,
                    message=f"{'Checked' if context.value == 'true' else 'Unchecked'} {context.purpose}"
                )
        
        return InteractionResult(
            success=False,
            message=f"Failed to {'check' if context.value == 'true' else 'uncheck'} {context.purpose}"
        )
    
    def _get_llm_guidance(self, context: InteractionContext) -> Dict[str, Any]:
        """Get LLM guidance for interaction"""
        prompt = f"""
        Analyze the following interaction:
        Element Type: {context.element_type}
        Action: {context.action}
        Purpose: {context.purpose}
        Value: {context.value}
        
        Generate selectors and steps for this interaction.
        Consider:
        1. Element type specific structure
        2. Common UI component patterns
        3. Hidden accessibility elements
        4. Fallback strategies
        
        Return a JSON object with:
        1. "selectors": array of CSS selectors to try
        2. "special_handling": (optional) array of special steps if needed
        3. "verification": (optional) verification steps
        """
        
        return self.llm_service.get_structured_guidance(prompt)
    
    def _retry_action(self, action_func, *args) -> bool:
        """Generic retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                result = action_func(*args)
                if result:
                    return True
            except Exception as e:
                self.logger.debug(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Action failed after {self.max_retries} attempts: {str(e)}")
                    return False
                self.page.wait_for_timeout(self.retry_delay)
        return False
    
    def _click_element(self, selector: str, purpose: str) -> bool:
        """Click an element"""
        element = self.page.locator(selector).first
        element.click()
        self.voice_service.speak(f"👆 Clicked {purpose}")
        return True
    
    def _type_text(self, selector: str, text: str, purpose: str) -> bool:
        """Type text into an element"""
        element = self.page.locator(selector).first
        element.fill(text)
        self.voice_service.speak(f"⌨️ Entered {purpose}")
        return True
    
    def _select_option(self, selector: str, option: str, purpose: str) -> bool:
        """Select an option from a dropdown"""
        element = self.page.locator(selector).first
        
        # Handle different types of dropdowns
        is_select = element.evaluate("el => el.tagName.toLowerCase() === 'select'")
        is_primeng = self._is_primeng_dropdown(selector)
        
        if is_select:
            # Handle standard HTML select
            element.select_option(label=option)
        elif is_primeng:
            # Handle PrimeNG dropdown
            self._handle_primeng_dropdown(selector, option)
        else:
            # Handle custom dropdown
            element.click()
            self.page.wait_for_timeout(500)
            option_selector = self._find_option_selector(option)
            if option_selector:
                self.page.locator(option_selector).click()
        
        self.voice_service.speak(f"📝 Selected {option} from {purpose}")
        return True
    
    def _hover_element(self, selector: str, purpose: str) -> bool:
        """Hover over an element"""
        element = self.page.locator(selector).first
        element.hover()
        self.voice_service.speak(f"🖱️ Hovering over {purpose}")
        return True
    
    def _toggle_checkbox(self, selector: str, action: str, purpose: str) -> bool:
        """Toggle a checkbox"""
        element = self.page.locator(selector).first
        current_state = element.is_checked()
        
        if action == "check" and not current_state:
            element.click()
        elif action == "uncheck" and current_state:
            element.click()
        
        self.voice_service.speak(f"✓ {action.capitalize()}ed {purpose}")
        return True
    
    def _is_primeng_dropdown(self, selector: str) -> bool:
        """Check if the element is a PrimeNG dropdown"""
        try:
            # Check for PrimeNG specific classes or attributes
            has_primeng_class = self.page.locator(selector).evaluate("""
                el => {
                    return el.classList.contains('p-dropdown') || 
                           el.classList.contains('p-dropdown-trigger') ||
                           el.closest('.p-dropdown') !== null;
                }
            """)
            return has_primeng_class
        except Exception:
            return False
    
    def _handle_primeng_dropdown(self, selector: str, option: str) -> bool:
        """Handle PrimeNG dropdown selection"""
        try:
            # Click to open dropdown
            dropdown_element = self.page.locator(selector).first
            dropdown_element.click()
            
            # Wait for dropdown panel to appear
            self.page.wait_for_selector('.p-dropdown-panel.p-component', state='visible', timeout=3000)
            
            # First check for filter in the dropdown panel
            filter_selector = '.p-dropdown-panel .p-dropdown-filter'
            has_filter = self.page.locator(filter_selector).count() > 0
            
            if has_filter:
                # Use filter to find option
                self.page.fill(filter_selector, option)
                self.page.wait_for_timeout(500)
            
            # Try to find and click the option
            option_selectors = [
                f".p-dropdown-panel .p-dropdown-item:text-is('{option}')",
                f".p-dropdown-panel .p-dropdown-item:text-contains('{option}')",
                f".p-dropdown-panel li:text-contains('{option}')"
            ]
            
            for option_selector in option_selectors:
                if self.page.locator(option_selector).count() > 0:
                    self.page.locator(option_selector).first.click()
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"PrimeNG dropdown error: {str(e)}")
            return False
    
    def _find_option_selector(self, option: str) -> Optional[str]:
        """Find selector for dropdown option"""
        guidance = self._get_llm_guidance(InteractionContext(
            purpose=f"find option {option}",
            element_type="option",
            action="find",
            value=option
        ))
        
        for selector in guidance.get("selectors", []):
            if self.page.locator(selector).count() > 0:
                return selector
        return None
    
    def _verify_selection(self, verification_steps: List[Dict], context: InteractionContext) -> bool:
        """Verify the selection was successful"""
        for step in verification_steps:
            try:
                if not self._execute_verification_step(step, context):
                    return False
            except Exception as e:
                self.logger.error(f"Verification failed: {str(e)}")
                return False
        return True
    
    def _execute_verification_step(self, step: Dict, context: InteractionContext) -> bool:
        """Execute a single verification step"""
        step_type = step.get("type")
        if step_type == "check_text":
            return self._verify_text(step["selector"], step["expected"])
        elif step_type == "check_value":
            return self._verify_value(step["selector"], step["expected"])
        elif step_type == "check_state":
            return self._verify_state(step["selector"], step["expected"])
        return False
    
    def _verify_text(self, selector: str, expected: str) -> bool:
        """Verify text content of an element"""
        try:
            element = self.page.locator(selector).first
            text = element.inner_text()
            return expected.lower() in text.lower()
        except Exception:
            return False
    
    def _verify_value(self, selector: str, expected: str) -> bool:
        """Verify value of an input element"""
        try:
            element = self.page.locator(selector).first
            value = element.input_value()
            return expected.lower() in value.lower()
        except Exception:
            return False
    
    def _verify_state(self, selector: str, expected: str) -> bool:
        """Verify state of an element"""
        try:
            element = self.page.locator(selector).first
            if expected.lower() == "checked":
                return element.is_checked()
            elif expected.lower() == "unchecked":
                return not element.is_checked()
            elif expected.lower() == "selected":
                return element.evaluate("el => el.selected")
            elif expected.lower() == "visible":
                return element.is_visible()
            elif expected.lower() == "enabled":
                return element.is_enabled()
            elif expected.lower() == "disabled":
                return not element.is_enabled()
            return False
        except Exception:
            return False
    
    def _select_product(self, product_name: str, should_check: bool) -> bool:
        """Handle product selection from the product list"""
        try:
            # Find the product container by its name
            product_selector = f"//div[contains(@class, 'wizard-card-checkbox-text1')]//div[contains(text(), '{product_name}')]/ancestor::div[contains(@class, 'wizard-card-checkbox-container')]"
            
            # Find the checkbox within the product container
            checkbox_selector = f"{product_selector}//div[contains(@class, 'p-checkbox')]"
            
            # Get the checkbox element
            checkbox = self.page.locator(checkbox_selector).first
            
            # Check if checkbox is already in desired state
            is_checked = checkbox.evaluate("""el => {
                return el.classList.contains('p-checkbox-checked')
            }""")
            
            # Only click if the current state doesn't match desired state
            if is_checked != should_check:
                checkbox.click()
                self.page.wait_for_timeout(500)  # Wait for any animations/updates
                
                # Verify the change
                new_state = checkbox.evaluate("""el => {
                    return el.classList.contains('p-checkbox-checked')
                }""")
                
                if new_state == should_check:
                    self.voice_service.speak(f"{'Selected' if should_check else 'Deselected'} {product_name}")
                    return True
            else:
                # Already in desired state
                self.voice_service.speak(f"{product_name} is already {'selected' if should_check else 'deselected'}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to select product {product_name}: {str(e)}")
            return False
    
    def _execute_step(self, step: Dict) -> None:
        """Execute a step from LLM guidance"""
        try:
            action = step.get("action")
            selector = step.get("selector")
            value = step.get("value")
            
            if action == "click" and selector:
                self.page.locator(selector).click()
                self.page.wait_for_timeout(500)
            elif action == "type" and selector and value:
                self.page.locator(selector).fill(value)
                self.page.wait_for_timeout(500)
            elif action == "wait" and selector:
                self.page.wait_for_selector(selector, state="visible", timeout=5000)
        except Exception as e:
            self.logger.error(f"Error executing step: {str(e)}")
