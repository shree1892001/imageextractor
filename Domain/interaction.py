"""
Web interaction logic for the Voice Assistant application.
"""
import logging
import re
from typing import Dict, List, Any, Optional

from Domain.models import InteractionContext, InteractionResult


class WebInteractor:
    """Reusable web interaction class"""
    
    def __init__(self, browser_service, llm_service, voice_service):
        """Initialize the web interactor"""
        self.browser_service = browser_service
        self.llm_service = llm_service
        self.voice_service = voice_service
        self.logger = logging.getLogger(__name__)
    
    def interact(self, context: InteractionContext) -> InteractionResult:
        """Perform an interaction with the web page"""
        return self.browser_service.interact(context)
    
    def click(self, purpose: str, selectors: Optional[List[str]] = None) -> InteractionResult:
        """Click on an element"""
        context = InteractionContext(
            purpose=purpose,
            element_type="button",
            action="click",
            selectors=selectors
        )
        return self.interact(context)
    
    def type_text(self, purpose: str, text: str, selectors: Optional[List[str]] = None) -> InteractionResult:
        """Type text into an input field"""
        context = InteractionContext(
            purpose=purpose,
            element_type="input",
            action="type",
            value=text,
            selectors=selectors
        )
        return self.interact(context)
    
    def select_option(self, purpose: str, option: str, selectors: Optional[List[str]] = None) -> InteractionResult:
        """Select an option from a dropdown"""
        context = InteractionContext(
            purpose=purpose,
            element_type="select",
            action="select",
            value=option,
            selectors=selectors
        )
        return self.interact(context)
    
    def hover(self, purpose: str, selectors: Optional[List[str]] = None) -> InteractionResult:
        """Hover over an element"""
        context = InteractionContext(
            purpose=purpose,
            element_type="element",
            action="hover",
            selectors=selectors
        )
        return self.interact(context)
    
    def check_checkbox(self, purpose: str, checked: bool = True, selectors: Optional[List[str]] = None) -> InteractionResult:
        """Check or uncheck a checkbox"""
        context = InteractionContext(
            purpose=purpose,
            element_type="checkbox",
            action="checkbox",
            value="true" if checked else "false",
            selectors=selectors
        )
        return self.interact(context)
    
    def select_product(self, product_name: str, should_check: bool = True) -> InteractionResult:
        """Select a product by name"""
        context = InteractionContext(
            purpose=f"{'Select' if should_check else 'Deselect'} {product_name}",
            element_type="checkbox",
            action="checkbox",
            value="true" if should_check else "false",
            product_name=product_name
        )
        return self.interact(context)
    
    def select_state(self, state_name: str) -> InteractionResult:
        """Select a state from a dropdown"""
        context = InteractionContext(
            purpose="state selection",
            element_type="dropdown",
            action="select",
            value=state_name
        )
        return self.interact(context)
    
    def handle_login(self, email: str, password: str) -> bool:
        """Handle login with email and password"""
        # Get page context
        page_context = self.browser_service.get_page_context()
        
        # Find email field
        email_selectors = self.llm_service.get_selectors("find email or username input field", page_context)
        email_result = False
        for selector in email_selectors:
            email_context = InteractionContext(
                purpose="email address",
                element_type="input",
                action="type",
                value=email,
                selectors=[selector]
            )
            result = self.interact(email_context)
            if result.success:
                email_result = True
                break
        
        # Find password field
        password_selectors = self.llm_service.get_selectors("find password input field", page_context)
        password_result = False
        for selector in password_selectors:
            password_context = InteractionContext(
                purpose="password",
                element_type="input",
                action="type",
                value=password,
                selectors=[selector]
            )
            result = self.interact(password_context)
            if result.success:
                password_result = True
                break
        
        # Find login button
        if email_result and password_result:
            login_button_selectors = self.llm_service.get_selectors("find login or sign in button", page_context)
            for selector in login_button_selectors:
                login_context = InteractionContext(
                    purpose="login button",
                    element_type="button",
                    action="click",
                    selectors=[selector]
                )
                result = self.interact(login_context)
                if result.success:
                    return True
            
            self.voice_service.speak("Filled login details but couldn't find login button")
            return False
        else:
            self.voice_service.speak("Could not find all required login fields")
            return False
    
    def handle_search(self, query: str) -> bool:
        """Handle search functionality"""
        # Get page context
        page_context = self.browser_service.get_page_context()
        
        # Find search field
        search_selectors = self.llm_service.get_selectors("find search input field", page_context)
        for selector in search_selectors:
            search_context = InteractionContext(
                purpose="search query",
                element_type="input",
                action="type",
                value=query,
                selectors=[selector]
            )
            result = self.interact(search_context)
            if result.success:
                # Press Enter to submit search
                self.browser_service.page.locator(selector).press("Enter")
                self.voice_service.speak(f"🔍 Searching for '{query}'")
                self.browser_service.page.wait_for_timeout(3000)
                return True
        
        self.voice_service.speak("Could not find search field")
        return False
    
    def handle_menu_navigation(self, menu_item: str, parent_menu: Optional[str] = None) -> bool:
        """Handle menu navigation"""
        # Get page context
        page_context = self.browser_service.get_page_context()
        
        if parent_menu:
            # Handle submenu navigation
            parent_selectors = self.llm_service.get_selectors(f"find menu item '{parent_menu}'", page_context)
            
            parent_found = False
            for selector in parent_selectors:
                hover_context = InteractionContext(
                    purpose=f"'{parent_menu}' menu",
                    element_type="menu",
                    action="hover",
                    selectors=[selector]
                )
                result = self.interact(hover_context)
                if result.success:
                    parent_found = True
                    break
            
            if not parent_found:
                self.voice_service.speak(f"Could not find parent menu '{parent_menu}'")
                return False
            
            # Get updated context after hovering
            updated_context = self.browser_service.get_page_context()
            submenu_selectors = self.llm_service.get_selectors(
                f"find submenu item '{menu_item}' under '{parent_menu}'",
                updated_context
            )
            
            for selector in submenu_selectors:
                click_context = InteractionContext(
                    purpose=f"submenu item '{menu_item}'",
                    element_type="menu",
                    action="click",
                    selectors=[selector]
                )
                result = self.interact(click_context)
                if result.success:
                    return True
            
            self.voice_service.speak(f"Could not find submenu item '{menu_item}' under '{parent_menu}'")
            return False
        else:
            # Handle direct menu item click
            menu_selectors = self.llm_service.get_selectors(f"find menu item '{menu_item}'", page_context)
            
            for selector in menu_selectors:
                click_context = InteractionContext(
                    purpose=f"menu item '{menu_item}'",
                    element_type="menu",
                    action="click",
                    selectors=[selector]
                )
                result = self.interact(click_context)
                if result.success:
                    return True
            
            self.voice_service.speak(f"Could not find menu item '{menu_item}'")
            return False
