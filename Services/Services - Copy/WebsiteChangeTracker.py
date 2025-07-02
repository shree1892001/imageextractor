import asyncio
import json
from datetime import datetime
from typing import List, Dict, Set, Optional
from playwright.async_api import async_playwright, Page
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *

class ElementState(BaseModel):
    selector: str
    exists: bool
    innerText: str = ""
    innerHTML: str = ""
    outerHTML: str = ""
    attributes: Dict[str, str] = {}
    styles: Dict[str, str] = {}
    boundingBox: Dict[str, float] = {}

class InitialSelector(BaseModel):
    selector: str
    initial_state: Dict
    last_check: datetime
    importance: float
    context: str
    change_threshold: float

class AIWebsiteMonitor:
    def __init__(self, api_key: str, output_file: str = "ai_website_changes.txt"):
        self.llm = GeminiModel("gemini-1.5-flash", api_key=api_key)
        self.system_prompt = """You are an AI agent specialized in website monitoring.
        Your task is to identify and track important elements on the page.
        Focus on elements that contain valuable information or are likely to change."""

        self.agent = Agent(model=self.llm, system_prompt=self.system_prompt, retries=3)
        self.output_file = output_file
        self.html_output_file = output_file.replace('.txt', '_html.txt')
        self.initial_selectors_file = output_file.replace('.txt', '_initial_selectors.json')
        self.initial_selectors: Dict[str, InitialSelector] = {}
        self.previous_state: Dict[str, ElementState] = {}
        self.change_history: List[Dict] = []
        self.initialize_files()

    def initialize_files(self):
        """Initialize output files and load initial selectors if they exist"""
        for file in [self.output_file, self.html_output_file]:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(f"=== Website Monitoring Log ===\n")
                f.write(f"Started: {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")

        try:
            with open(self.initial_selectors_file, 'r') as f:
                stored_data = json.load(f)
                self.initial_selectors = {
                    k: InitialSelector(**v) for k, v in stored_data.items()
                }
        except FileNotFoundError:
            self.initial_selectors = {}

    async def validate_selector(self, page: Page, selector: str) -> bool:
        """Validate if a selector exists and is specific enough"""
        try:
            elements = await page.query_selector_all(selector)
            return 0 < len(elements) <= 5
        except Exception:
            return False

    async def extract_element_state(self, page: Page, selector: str) -> ElementState:
        """Extract current state of an element"""
        try:
            element = await page.query_selector(selector)
            if not element:
                return ElementState(selector=selector, exists=False)

            data = await element.evaluate("""element => ({
                innerText: element.innerText,
                innerHTML: element.innerHTML,
                outerHTML: element.outerHTML,
                attributes: Object.fromEntries(
                    [...element.attributes].map(attr => [attr.name, attr.value])
                ),
                styles: (() => {
                    const computed = window.getComputedStyle(element);
                    return {
                        visibility: computed.visibility,
                        display: computed.display,
                        position: computed.position,
                        backgroundColor: computed.backgroundColor,
                        color: computed.color,
                        fontSize: computed.fontSize
                    };
                })(),
                boundingBox: element.getBoundingClientRect().toJSON()
            })""")

            return ElementState(
                selector=selector,
                exists=True,
                **data
            )

        except Exception as e:
            return ElementState(selector=selector, exists=False)

    async def analyze_element_context(self, page: Page, selector: str) -> tuple[str, float, float]:
        """Analyze element context and determine importance and change threshold"""
        try:
            element = await page.query_selector(selector)
            if not element:
                return "Element not found", 0.5, 0.3

            content = await element.inner_text()
            surrounding = await page.evaluate(f"""
                selector => {{
                    const element = document.querySelector(selector);
                    const parent = element.parentElement;
                    return parent ? parent.innerText : '';
                }}
            """, selector)

            prompt = f"""Analyze this element on the webpage:

            Selector: {selector}
            Content: {content}
            Surrounding Context: {surrounding}

            Provide three values in this exact format:
            Description: [one sentence describing the element's purpose]
            Importance (0-1): [number indicating how critical this element is]
            Change Threshold (0-1): [number indicating how significant a change should be to report it]"""

            response = await self.agent.run(prompt)
            lines = response.data.strip().split('\n')

            description = lines[0].split('Description: ')[1]
            importance = float(lines[1].split('Importance (0-1): ')[1])
            threshold = float(lines[2].split('Change Threshold (0-1): ')[1])

            return description, importance, threshold

        except Exception as e:
            print(f"Error analyzing element context: {str(e)}")
            return "Error analyzing element", 0.5, 0.3

    async def discover_selectors(self, page: Page) -> Set[str]:
        """Use AI to discover important elements and generate selectors"""
        try:
            page_content = await page.content()

            prompt = f"""Analyze this HTML and provide CSS selectors for important elements.
            For each important element you find, write one line as: "selector: reason for tracking"

            Focus on elements like:
            - Dynamic content areas
            - Prices and product information
            - Status indicators
            - Navigation elements
            - Main content sections
            - Interactive elements
            - Forms and inputs

            HTML content:
            {page_content}"""

            response = await self.agent.run(prompt)
            new_selectors = set()

            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write("\n=== Newly Discovered Selectors ===\n")

                for line in response.data.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        selector, explanation = line.split(':', 1)
                        selector = selector.strip()
                        explanation = explanation.strip()

                        f.write(f"Selector: {selector}\n")
                        f.write(f"Reason: {explanation}\n")
                        f.write("-" * 40 + "\n")

                        new_selectors.add(selector)

            return new_selectors

        except Exception as e:
            print(f"Error discovering selectors: {str(e)}")
            return set()

    async def store_initial_selector(self, page: Page, selector: str):
        """Store initial state of a selector with context"""
        try:
            element_state = await self.extract_element_state(page, selector)
            if not element_state.exists:
                return

            context, importance, threshold = await self.analyze_element_context(page, selector)

            initial_selector = InitialSelector(
                selector=selector,
                initial_state=element_state.dict(),
                last_check=datetime.now().isoformat(),
                importance=importance,
                context=context,
                change_threshold=threshold
            )

            self.initial_selectors[selector] = initial_selector

            with open(self.initial_selectors_file, 'w') as f:
                json.dump(
                    {k: v.dict() for k, v in self.initial_selectors.items()},
                    f,
                    indent=2
                )

            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== Initial Selector Stored ===\n")
                f.write(f"Selector: {selector}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Importance: {importance}\n")
                f.write(f"Change Threshold: {threshold}\n")
                f.write("-" * 40 + "\n")

        except Exception as e:
            print(f"Error storing initial selector {selector}: {str(e)}")

    async def analyze_element_changes(self, page: Page, selector: str, current_state: ElementState, initial_data: InitialSelector) -> Optional[Dict]:
        """Use AI to analyze if changes in an element are significant"""
        try:
            if not current_state.exists:
                return {
                    "type": "disappeared",
                    "selector": selector,
                    "importance": initial_data.importance
                }

            prompt = f"""Compare these two states of a webpage element and determine if there are meaningful changes:

            Initial State:
            {json.dumps(initial_data.initial_state, indent=2)}

            Current State:
            {json.dumps(current_state.dict(), indent=2)}

            Element Context: {initial_data.context}
            Importance: {initial_data.importance}
            Change Threshold: {initial_data.change_threshold}

            Analyze if there are significant changes that exceed the change threshold.
            Return in this format:
            Changed: [yes/no]
            Significance (0-1): [number]
            Description: [what changed and why it matters]"""

            response = await self.agent.run(prompt)
            lines = response.data.strip().split('\n')

            changed = lines[0].split('Changed: ')[1].lower() == 'yes'
            significance = float(lines[1].split('Significance (0-1): ')[1])
            description = lines[2].split('Description: ')[1]

            if changed and significance > initial_data.change_threshold:
                return {
                    "type": "significant_change",
                    "selector": selector,
                    "significance": significance,
                    "description": description,
                    "importance": initial_data.importance
                }

            return None

        except Exception as e:
            print(f"Error analyzing changes: {str(e)}")
            return None

    async def log_html(self, state: ElementState):
        """Log HTML state"""
        with open(self.html_output_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] {state.selector}\n")
            f.write("-" * 40 + "\n")

            if state.exists:
                f.write(f"HTML Content:\n{state.outerHTML}\n")
            else:
                f.write("Element not found\n")

            f.write("-" * 40 + "\n")

    async def log_significant_changes(self, changes: List[Dict]):
        """Log significant changes detected by AI"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] Significant Changes Detected\n")
            f.write("-" * 40 + "\n")

            for change in changes:
                f.write(f"Selector: {change['selector']}\n")
                f.write(f"Type: {change['type']}\n")
                if 'significance' in change:
                    f.write(f"Significance: {change['significance']}\n")
                if 'description' in change:
                    f.write(f"Description: {change['description']}\n")
                f.write(f"Importance: {change['importance']}\n")
                f.write("-" * 40 + "\n")

    async def track_changes(self, page: Page):
        """Main tracking function with AI-powered change detection"""
        new_selectors = await self.discover_selectors(page)

        for selector in new_selectors:
            if await self.validate_selector(page, selector) and selector not in self.initial_selectors:
                await self.store_initial_selector(page, selector)

        significant_changes = []

        for selector, initial_data in self.initial_selectors.items():
            current_state = await self.extract_element_state(page, selector)
            await self.log_html(current_state)

            change = await self.analyze_element_changes(
                page,
                selector,
                current_state,
                initial_data
            )

            if change:
                significant_changes.append(change)
                initial_data.last_check = datetime.now()

        if significant_changes:
            await self.log_significant_changes(significant_changes)
            self.change_history.append({
                "timestamp": datetime.now().isoformat(),
                "changes": significant_changes
            })

    async def monitor_website(self, url: str, interval: int = 30):
        """Main monitoring loop"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            try:
                print(f"🔍 Starting monitoring of {url}")
                print(f"📝 Logging to: {self.output_file}")

                await page.goto(url)
                print("⚡ Initial scan...")
                await self.track_changes(page)

                print(f"⏳ Monitoring every {interval} seconds...")
                while True:
                    await asyncio.sleep(interval)
                    await page.reload()
                    await self.track_changes(page)

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"❌ {error_msg}")
                with open(self.output_file, 'a') as f:
                    f.write(f"\n[ERROR] {error_msg}\n")
            finally:
                await browser.close()

async def main():
    monitor = AIWebsiteMonitor(
        api_key=API_KEY
    )
    website_url = NC_URL
    await monitor.monitor_website(website_url, interval=30)

if __name__ == "__main__":
    asyncio.run(main())