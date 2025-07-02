from lavague.core import WorldModel, ActionEngine
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from lavague.contexts.gemini import GeminiContext
from lavague.actions import NavigationActions
from Common.constants import *

class CustomWebAgent(WebAgent):
    def __init__(self, world_model, action_engine):
        super().__init__(world_model, action_engine)

        self.register_actions(NavigationActions())

    def navigate_to(self, url):
        """Custom navigation method"""
        return self.action_engine.execute_action("navigate", {"url": url})

context = GeminiContext(api_key=API_KEY)

selenium_driver = SeleniumDriver(
    headless=False,
    navigation_timeout=30
)

action_engine = ActionEngine.from_context(
    context=context,
    driver=selenium_driver,
    supported_actions=["navigate", "click", "type", "wait"]
)

world_model = WorldModel.from_context(context)

agent = CustomWebAgent(world_model, action_engine)

try:
    instructions = AUTOMATION_TASK
    agent.run(instructions)
except Exception as e:
    print(f"Automation Error: {str(e)}")

    selenium_driver.quit()