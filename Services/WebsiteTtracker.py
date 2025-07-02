import asyncio
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple
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
    importance: float = 0.0
    category: str = ""

class ElementMetadata(BaseModel):
    importance: float = 0.0
    category: str = "unknown"
    reason: str = ""
    update_frequency: str = "unknown"
    last_validated: str = ""
    initial_state: Dict = {}

class AIWebsiteMonitor:
    def __init__(self, api_key: str, output_file: str = "ai_website_changes.txt"):
        self.llm = GeminiModel("gemini-1.5-flash", api_key=api_key)
        self.agent = Agent(model=self.llm, system_prompt="""
        You are an AI agent specialized in website monitoring and DOM analysis.
        Compare the current state with the initial state of elements and identify:
        1. New elements that should be monitored
        2. Changes in existing elements
        3. Elements that are no longer present
        """, retries=3)
        self.output_file = output_file
        self.state_file = output_file.replace('.txt', '_state.json')
        self.selectors: Dict[str, ElementMetadata] = {}
        self.initial_state: Dict[str, ElementState] = {}
        self.previous_state: Dict[str, ElementState] = {}
        self.change_history: List[Dict] = []
        self.initialize_files()

    def initialize_files(self):
        try:
            with open(self.state_file, 'r') as f:
                saved_state = json.load(f)
                self.initial_state = {k: ElementState(**v) for k, v in saved_state.get('initial_state', {}).items()}
                self.selectors = {k: ElementMetadata(**v) for k, v in saved_state.get('selectors', {}).items()}
        except FileNotFoundError:
            self.initial_state = {}
            self.selectors = {}

    async def save_state(self):
        state_data = {
            'initial_state': {k: v.dict() for k, v in self.initial_state.items()},
            'selectors': {k: v.dict() for k, v in self.selectors.items()},
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    async def track_changes(self, page: Page):
        """Enhanced change tracking with flexible AI analysis"""

        new_elements = await self.analyze_page_structure(page)

        for selector, metadata in new_elements:
            if selector not in self.selectors and await self.validate_selector(page, selector):
                self.selectors[selector] = metadata

                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== New Element Detected ===\n")
                    f.write(f"Selector: {selector}\n")
                    f.write(f"Importance: {metadata.importance}\n")
                    f.write(f"Category: {metadata.category}\n")
                    f.write(f"Reason: {metadata.reason}\n")
                    f.write(f"Update Frequency: {metadata.update_frequency}\n")
                    f.write("-" * 40 + "\n")

        current_state = {}
        changes = []

        for selector, metadata in self.selectors.items():
            element_state = await self.extract_element_state(page, selector)
            current_state[selector] = element_state

            updated_metadata = await self.update_element_metadata(page, selector, element_state)
            if updated_metadata:
                self.selectors[selector] = updated_metadata

            if selector in self.previous_state:
                selector_changes = self.detect_changes(
                    self.previous_state[selector],
                    element_state
                )

                for change in selector_changes:
                    change["importance"] = self.selectors[selector].importance
                    change["category"] = self.selectors[selector].category
                    change["update_frequency"] = self.selectors[selector].update_frequency

                changes.extend(selector_changes)

            await self.log_html(element_state)

        if changes:
            await self.log_changes(changes)

        self.previous_state = current_state
        self.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "changes": changes
        })
    async def extract_element_state(self, page: Page, selector: str) -> ElementState:
        try:
            element = await page.query_selector(selector)
            if not element:
                return ElementState(selector=selector, exists=False)
            return ElementState(
                selector=selector,
                exists=True,
                innerText=await element.inner_text(),
                innerHTML=await element.inner_html(),
                outerHTML=await element.outer_html(),
                attributes=await element.evaluate('el => Object.fromEntries(el.attributes)')
                if element else {},
                styles=await element.evaluate('el => Object.fromEntries(getComputedStyle(el))')
                if element else {},
                boundingBox=await element.bounding_box() or {}
            )
        except:
            return ElementState(selector=selector, exists=False)

    async def initialize_monitoring(self, page: Page):
        initial_elements = await self.analyze_page_structure(page)
        for selector, metadata in initial_elements:
            if await self.validate_selector(page, selector):
                element_state = await self.extract_element_state(page, selector)
                self.initial_state[selector] = element_state
                metadata.initial_state = element_state.dict()
                self.selectors[selector] = metadata
        await self.save_state()

    async def analyze_page_structure(self, page: Page) -> List[Tuple[str, ElementMetadata]]:
        try:
            response = await self.agent.run(f"""Analyze this HTML structure and identify important elements to monitor.
            {await page.content()}""")
            return self.parse_selector_info(response.data)
        except:
            return []

    def parse_selector_info(self, text: str) -> List[Tuple[str, ElementMetadata]]:
        selector_info = []
        sections = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
        for section in sections:
            selector_match = re.search(r'(?:selector:\s*|[\'\"])((?:\w+|\#\w+|\.\w+)(?:\s*[\>\+\~]\s*(?:\w+|\#\w+|\.\w+))*)[\'\"]?', section)
            if not selector_match:
                continue
            selector = selector_match.group(1).strip()
            metadata = ElementMetadata(
                importance=self.parse_importance(section),
                category=self.extract_category(section),
                reason=section.strip(),
                update_frequency=self.parse_update_frequency(section),
                last_validated=datetime.now().isoformat()
            )
            selector_info.append((selector, metadata))
        return selector_info

    async def monitor_website(self, url: str, interval: int = 30):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            try:
                await page.goto(url)
                if not self.initial_state:
                    await self.initialize_monitoring(page)
                while True:
                    await asyncio.sleep(interval)
                    await page.reload()
                    await self.track_changes(page)
            finally:
                await browser.close()

async def main():
    monitor = AIWebsiteMonitor(api_key=API_KEY)
    await monitor.monitor_website(NC_URL, interval=30)

if __name__ == "__main__":
    asyncio.run(main())
