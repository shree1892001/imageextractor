from enum import Enum
from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

class ElementType(Enum):
    DROPDOWN = "dropdown"
    BUTTON = "button"
    INPUT = "input"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    LINK = "link"

@dataclass
class ElementContext:
    """Base context for UI elements with common attributes"""
    purpose: str
    surrounding_text: str
    element_type: ElementType
    page_content: str = ""
    url: str = ""
    title: str = ""

class UIElement(ABC):
    """Abstract base class for all UI elements"""
    def __init__(self, page, llm_selector, voice_engine):
        self.page = page
        self.llm_selector = llm_selector
        self.voice_engine = voice_engine

    @abstractmethod
    async def interact(self, context: ElementContext) -> bool:
        """Main interaction method to be implemented by specific elements"""
        pass

    async def _try_selectors(self, selectors: List[str], action: str) -> bool:
        """Common method to try multiple selectors"""
        for selector in selectors:
            try:
                element = self.page.locator(selector)
                if await element.count() > 0:
                    await element.click()
                    return True
            except Exception as e:
                continue
        return False

    async def _verify_element_exists(self, selector: str) -> bool:
        """Verify if element exists on page"""
        try:
            element = self.page.locator(selector)
            return await element.count() > 0
        except:
            return False