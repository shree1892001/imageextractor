import asyncio
import json
import hashlib
from typing import Dict, Any, List, Set
from datetime import datetime
from playwright.async_api import async_playwright, Page, ElementHandle
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from Common.constants import *

class WebsiteChangeTracker:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = """
        You are an intelligent web analyzer that specializes in:
        1. Detecting and analyzing all interactive elements on a webpage
        2. Understanding the semantic meaning and purpose of each element
        3. Identifying changes in website structure and layout
        4. Maintaining a robust selector strategy for each element

        For each element, provide:
        - Multiple reliable selectors (CSS, XPath, etc.)
        - The element's role and purpose
        - Any relevant attributes or properties
        - Relationships with other elements

        Output as detailed JSON with clear element categorization.
        """
        self.agent = Agent(model=self.llm, system_prompt=self.system_prompt, retries=3)
        self.previous_state = {}
        self.selector_history = {}

    async def capture_page_state(self, page: Page) -> Dict[str, Any]:
        """Captures the current state of all interactive elements on the page"""
        elements_data = {}

        elements = await page.query_selector_all(
            'button, input, select, textarea, a[href], [role="button"], [role="link"], [role="textbox"]'
        )

        for element in elements:
            try:

                selectors = await self._generate_selectors(page, element)

                properties = await self._get_element_properties(element)

                element_id = await self._generate_element_id(properties)

                element_analysis = await self._analyze_element(properties)

                elements_data[element_id] = {
                    "selectors": selectors,
                    "properties": properties,
                    "analysis": element_analysis,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Error processing element: {str(e)}")

        return elements_data

    async def _generate_selectors(self, page: Page, element: ElementHandle) -> List[str]:
        """Generates multiple reliable selectors for an element"""
        selectors = []

        props = await element.evaluate("""element => {
            return {
                id: element.id,
                name: element.name,
                className: element.className,
                tagName: element.tagName,
                type: element.type,
                placeholder: element.placeholder,
                ariaLabel: element.getAttribute('aria-label'),
                dataTestId: element.getAttribute('data-testid'),
                text: element.textContent
            }
        }""")

        if props['id']:
            selectors.append(f"#{props['id']}")
        if props['dataTestId']:
            selectors.append(f"[data-testid='{props['dataTestId']}']")
        if props['name']:
            selectors.append(f"{props['tagName'].lower()}[name='{props['name']}']")

        try:
            xpath = await page.evaluate("""element => {
                function getXPath(element) {
                    if (element.id)
                        return `//*[@id="${element.id}"]`;
                    if (element === document.body)
                        return '/html/body';

                    let ix = 0;
                    let siblings = element.parentNode.childNodes;

                    for (let i = 0; i < siblings.length; i++) {
                        let sibling = siblings[i];
                        if (sibling === element)
                            return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        if (sibling.nodeType === 1 && sibling.tagName === element.tagName)
                            ix++;
                    }
                }
                return getXPath(arguments[0]);
            }""", element)
            selectors.append(xpath)
        except Exception:
            pass

        return selectors

    async def _get_element_properties(self, element: ElementHandle) -> Dict[str, Any]:
        """Extracts relevant properties from an element"""
        return await element.evaluate("""element => {
            const style = window.getComputedStyle(element);
            return {
                tagName: element.tagName,
                id: element.id,
                className: element.className,
                type: element.type,
                value: element.value,
                placeholder: element.placeholder,
                ariaLabel: element.getAttribute('aria-label'),
                isVisible: style.display !== 'none' && style.visibility !== 'hidden',
                position: {
                    x: element.getBoundingClientRect().x,
                    y: element.getBoundingClientRect().y
                },
                dimensions: {
                    width: element.offsetWidth,
                    height: element.offsetHeight
                },
                attributes: Array.from(element.attributes).map(attr => ({
                    name: attr.name,
                    value: attr.value
                }))
            };
        }""")

    async def _generate_element_id(self, properties: Dict) -> str:
        """Generates a unique identifier for an element based on its properties"""
        relevant_props = {
            'tagName': properties['tagName'],
            'id': properties['id'],
            'className': properties['className'],
            'type': properties['type'],
            'position': properties['position']
        }
        return hashlib.md5(json.dumps(relevant_props).encode()).hexdigest()

    async def _analyze_element(self, properties: Dict) -> Dict[str, Any]:
        """Uses AI to analyze the element's purpose and role"""
        prompt = f"""
        Analyze this web element and determine its purpose and role:
        {json.dumps(properties, indent=2)}

        Provide:
        1. The most likely purpose of this element
        2. Its role in the user interface
        3. Any potential alternative ways to identify this element
        4. Suggested backup selectors
        """

        result = await self.agent.run(prompt)
        print(result.data)
        repsones= json.loads(result.data.strip().replace("```json", "").replace("```", "").strip())
        return repsones

    async def track_changes(self, page: Page) -> Dict[str, Any]:
        """Tracks changes in the website structure and elements"""
        current_state = await self.capture_page_state(page)
        changes = {
            'new_elements': {},
            'removed_elements': {},
            'modified_elements': {},
            'timestamp': datetime.now().isoformat()
        }

        if self.previous_state:

            for element_id, element_data in current_state.items():
                if element_id not in self.previous_state:
                    changes['new_elements'][element_id] = element_data
                elif element_data != self.previous_state[element_id]:
                    changes['modified_elements'][element_id] = {
                        'current': element_data,
                        'previous': self.previous_state[element_id]
                    }

            for element_id in self.previous_state:
                if element_id not in current_state:
                    changes['removed_elements'][element_id] = self.previous_state[element_id]

        self.previous_state = current_state

        for element_id, element_data in current_state.items():
            if element_id not in self.selector_history:
                self.selector_history[element_id] = []
            self.selector_history[element_id].append({
                'selectors': element_data['selectors'],
                'timestamp': element_data['timestamp']
            })

        return changes

    async def automate_login(self):
        """Enhanced login automation with change tracking"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            await page.goto(NC_URL)
            print("🔍 Analyzing initial page state...")

            initial_state = await self.track_changes(page)

            login_steps = ['username', 'password', 'login_button', 'otp_selection', 'otp_input']

            for step in login_steps:
                print(f"👉 Executing step: {step}")

                pre_action_state = await self.track_changes(page)

                if step == 'username':
                    await self._fill_field(page, QA_USERNAME, 'username')
                elif step == 'password':
                    await self._fill_field(page, QA_PASSWORD, 'password')
                elif step == 'login_button':
                    await self._click_button(page, 'login')
                elif step == 'otp_selection':
                    await asyncio.sleep(3)
                    await self._click_button(page, 'send_otp_email')
                elif step == 'otp_input':
                    await self._wait_for_otp(page)

                post_action_state = await self.track_changes(page)

                print(f"📊 Analyzing changes after {step}...")
                self._analyze_state_changes(pre_action_state, post_action_state)

                await asyncio.sleep(2)

            final_state = await self.track_changes(page)

            self._save_session_data({
                'initial_state': initial_state,
                'final_state': final_state,
                'selector_history': self.selector_history
            })

            await browser.close()

    async def _fill_field(self, page: Page, value: str, field_type: str):
        """Enhanced field filling with dynamic selector generation"""
        current_state = await self.capture_page_state(page)
        relevant_elements = self._find_relevant_elements(current_state, field_type)

        for element_id, element_data in relevant_elements.items():
            for selector in element_data['selectors']:
                try:
                    await page.fill(selector, value)
                    print(f"✅ Filled {field_type} using selector: {selector}")
                    return
                except Exception:
                    continue
        print(f"❌ Failed to fill {field_type}")

    async def _click_button(self, page: Page, button_type: str):
        """Enhanced button clicking with dynamic selector generation"""
        current_state = await self.capture_page_state(page)
        relevant_elements = self._find_relevant_elements(current_state, button_type)

        for element_id, element_data in relevant_elements.items():
            for selector in element_data['selectors']:
                try:
                    await page.click(selector)
                    print(f"✅ Clicked {button_type} using selector: {selector}")
                    return
                except Exception:
                    try:
                        await page.evaluate(f"document.querySelector('{selector}').click();")
                        print(f"✅ Clicked {button_type} using JS and selector: {selector}")
                        return
                    except Exception:
                        continue
        print(f"❌ Failed to click {button_type}")

    async def _wait_for_otp(self, page: Page):
        """Enhanced OTP field detection"""
        print("⏳ Waiting for OTP field...")
        current_state = await self.capture_page_state(page)
        relevant_elements = self._find_relevant_elements(current_state, 'otp')

        for element_id, element_data in relevant_elements.items():
            for selector in element_data['selectors']:
                try:
                    await page.wait_for_selector(selector, timeout=120000)
                    print("📩 OTP Field Detected!")
                    return
                except Exception:
                    continue
        print("❌ Failed to detect OTP field")

    def _find_relevant_elements(self, state: Dict[str, Any], element_type: str) -> Dict[str, Any]:
        """Finds elements relevant to the specified type based on AI analysis"""
        relevant_elements = {}
        for element_id, element_data in state.items():
            if element_type.lower() in str(element_data['analysis']).lower():
                relevant_elements[element_id] = element_data
        return relevant_elements

    def _analyze_state_changes(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]):
        """Analyzes changes between states"""
        changes = {
            'new_elements': len(post_state.get('new_elements', {})),
            'removed_elements': len(post_state.get('removed_elements', {})),
            'modified_elements': len(post_state.get('modified_elements', {}))
        }
        print(f"📊 Changes detected: {json.dumps(changes, indent=2)}")

    def _save_session_data(self, data: Dict[str, Any]):
        """Saves session data to a file"""
        filename = f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Session data saved to {filename}")

async def main():
    api_key = API_KEY
    tracker = WebsiteChangeTracker(api_key)
    await tracker.automate_login()

if __name__ == "__main__":
    asyncio.run(main())