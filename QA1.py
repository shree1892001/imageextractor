from pydantic_ai import Agent
from playwright.async_api import async_playwright
from typing import Dict, Any
import asyncio
from Common.constants import *


class BrowserAgent(Agent):
    async def run(self) -> Dict[str, Any]:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()

        try:
            steps = [
                {"action": "goto", "url": QA_URL},
                {"action": "fill", "selector": "#username", "value": QA_USERNAME},
                {"action": "fill", "selector": "#password", "value": QA_PAss},
                {"action": "click", "selector": "#login-button"},
                {"action": "waitFor", "selector": "#otp-button"},
                {"action": "click", "selector": "#otp-button"},
                {"action": "waitFor", "selector": "#otp-input", "timeout": 120000},
                {"action": "waitFor", "selector": ".dashboard-container", "timeout": 30000}
            ]

            for step in steps:
                if step["action"] == "goto":
                    await page.goto(step["url"])
                elif step["action"] == "fill":
                    await page.fill(step["selector"], step["value"])
                elif step["action"] == "click":
                    await page.click(step["selector"])
                elif step["action"] == "waitFor":
                    timeout = step.get("timeout", 30000)
                    await page.wait_for_selector(step["selector"], timeout=timeout)

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            await browser.close()
            await playwright.stop()


async def main():
    agent = BrowserAgent()
    result = await agent.run()
    print("Success" if result["success"] else f"Failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())