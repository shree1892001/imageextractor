from enum import Enum
from dataclasses import dataclass
from elements import ElementContext, UIElement
from typing import Optional,List

class DropdownType(Enum):
    STATE = "state"
    ENTITY = "entity"
    GENERIC = "generic"

@dataclass
class DropdownContext(ElementContext):
    """Specific context for dropdowns with additional attributes"""
    dropdown_type: DropdownType = DropdownType.GENERIC

class Dropdown(UIElement):
    """Reusable dropdown component with specific dropdown functionality"""
    
    def __init__(self, page, llm_selector, voice_engine):
        super().__init__(page, llm_selector, voice_engine)
        self.type_verifiers = {
            DropdownType.STATE: self._is_state_dropdown,
            DropdownType.ENTITY: self._is_entity_dropdown
        }

    def _is_state_dropdown(self, text: str) -> bool:
        keywords = ["state of formation", "formation state", "state", "location"]
        return any(keyword in text.lower() for keyword in keywords)

    def _is_entity_dropdown(self, text: str) -> bool:
        keywords = ["entity type", "business type", "company type", "organization type"]
        return any(keyword in text.lower() for keyword in keywords)

    async def _verify_dropdown_type(self, selector: str) -> Optional[DropdownType]:
        try:
            element = self.page.locator(selector)
            if await element.count() > 0:
                surrounding_text = await self._get_surrounding_text(element)
                
                for dropdown_type, verifier in self.type_verifiers.items():
                    if verifier(surrounding_text):
                        return dropdown_type
            return None
        except Exception:
            return None

    async def _get_surrounding_text(self, element) -> str:
        """Get surrounding text for context"""
        return await element.evaluate("""el => {
            const getParentText = (element, depth = 3) => {
                if (depth === 0 || !element) return '';
                const text = element.textContent || '';
                return element.parentElement ? 
                    text + ' ' + getParentText(element.parentElement, depth - 1) : 
                    text;
            };
            return getParentText(el);
        }""")

    async def interact(self, context: DropdownContext) -> bool:
        """Main interaction method for dropdowns"""
        try:
            selectors = await self._get_dropdown_selectors(context)
            return await self._try_dropdown_interaction(selectors, context)
        except Exception as e:
            print(f"Dropdown interaction error: {e}")
            return False

    async def _get_dropdown_selectors(self, context: DropdownContext) -> List[str]:
        """Get selectors using LLM"""
        prompt = self._build_selector_prompt(context)
        return self.llm_selector.generate_selectors(prompt)

    def _build_selector_prompt(self, context: DropdownContext) -> str:
        """Build prompt for LLM"""
        return f"""
        Generate selectors for a dropdown with these characteristics:
        - Type: {context.dropdown_type.value} dropdown
        - Purpose: {context.purpose}
        - Surrounding text: {context.surrounding_text}
        - Should specifically look for {context.dropdown_type.value}-related labels and attributes
        - Consider aria-labels and data attributes

        Current page title: {context.title}

        Respond with JSON array of selectors, ordered by specificity.
        """