import asyncio
import json
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright, Page, ElementHandle
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *


class LLCRegistrationAgent:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = GeminiModel(model, api_key=api_key)
        # Updated system prompt to enforce JSON structure

        self.system_prompt = '''You are an AI assistant specializing in web automation and form detection.
        Your task is to analyze webpage elements and identify relevant fields accurately.

        IMPORTANT: You must ALWAYS return your response as a valid JSON object with the following structure:
        {
            "elements": {
                "elementKey": {
                    "primary_selector": "css selector string",
                    "alternate_selectors": ["selector1", "selector2"],
                    "element_type": "input/button/select/etc",
                    "confidence_score": 0.95
                }
            }
        }

        Guidelines for element detection:
        1. Look for elements based on multiple attributes: id, name, class, placeholder, aria-label
        2. Consider text content of nearby labels and parent elements
        3. Prioritize unique identifiers over generic selectors
        4. Return CSS selectors that are specific yet resilient
        5. For forms, detect both visible and hidden fields
        6. Include alternative selectors when available

        REMEMBER: Your response must be a properly formatted JSON object as shown above.
        Do not include any explanatory text outside the JSON structure.'''
        self.agent = Agent(model=self.llm, system_prompt=self.system_prompt, retries=3)

    async def validate_json_response(self, response: str) -> Dict:
        """Validates and cleans AI response to ensure valid JSON."""
        try:
            # Try to find JSON content if there's additional text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = response[json_start:json_end]
                return json.loads(json_content)
            return {}
        except json.JSONDecodeError:
            print("Invalid JSON response from AI")
            return {}

    async def detect_elements(self, page: Page, task: str) -> Dict[str, Any]:
        """Enhanced element detection with strict JSON validation."""
        page_content = await self.get_page_content(page)

        prompt = f"""
        Task: {task}

        Current URL: {page_content['url']}
        Page Title: {page_content['title']}

        Available Form Elements:
        {json.dumps(page_content['formElements'], indent=2)}

        Visible Text Content:
        {json.dumps(page_content['visibleText'], indent=2)}

        IMPORTANT: Return a valid JSON object following the required structure.
        Do not include any text outside the JSON object.
        """

        result = await self.agent.run(prompt)
        if result and result.data:
            # Handle both string and list response formats
            response_text = result.data[0] if isinstance(result.data, list) else result.data
            validated_json = await self.validate_json_response(str(response_text))

            # Ensure response has expected structure
            if 'elements' in validated_json:
                return validated_json['elements']
            return {}
        return {}

    async def handle_login(self, page: Page, credentials: Dict) -> bool:
        """Enhanced login handling with strict JSON response parsing."""
        try:
            login_elements = await self.detect_elements(
                page,
                '''Identify login form elements. Return JSON with these keys:
                - "login_trigger" for login button/link
                - "username" for username field
                - "password" for password field
                - "submit" for submit button'''
            )

            if not login_elements:
                print("Failed to detect login elements")
                return False

            # Handle login button
            if login_trigger := login_elements.get('login_trigger'):
                if trigger_element := await self.verify_element(page, login_trigger):
                    await trigger_element.click()
                    await page.wait_for_load_state("networkidle")

            # Fill credentials
            if username_info := login_elements.get('username'):
                if username_field := await self.verify_element(page, username_info):
                    await username_field.fill(credentials['username'])

            if password_info := login_elements.get('password'):
                if password_field := await self.verify_element(page, password_info):
                    await password_field.fill(credentials['password'])

            # Submit
            if submit_info := login_elements.get('submit'):
                if submit_button := await self.verify_element(page, submit_info):
                    await submit_button.click()
                    await page.wait_for_load_state("networkidle")

                    await page.wait_for_timeout(2000)
                    error_elements = await page.query_selector_all('[class*="error"], [class*="alert"]')
                    return not bool(error_elements)

            return False

        except Exception as e:
            print(f"Login failed: {e}")
            return False

    async def fill_form_data(self, page: Page, form_data: Dict) -> int:
        """Enhanced form filling with strict JSON validation."""
        fields_filled = 0

        async def fill_nested_data(data: Dict, parent_key: str = ""):
            nonlocal fields_filled

            for key, value in data.items():
                current_key = f"{parent_key}_{key}" if parent_key else key

                if isinstance(value, dict):
                    await fill_nested_data(value, current_key)
                else:
                    field_elements = await self.detect_elements(
                        page,
                        f'''Identify form field for {current_key}.
                        Field should accept value of type: {type(value).__name__}.
                        Return JSON with "{current_key}" as the main key.'''
                    )

                    if field_info := field_elements.get(current_key):
                        if field_element := await self.verify_element(page, field_info):
                            try:
                                element_type = field_info.get('element_type', 'text')

                                if element_type == 'select':
                                    await field_element.select_option(value)
                                elif element_type == 'checkbox':
                                    if value:
                                        await field_element.check()
                                    else:
                                        await field_element.uncheck()
                                else:
                                    await field_element.fill(str(value))
                                fields_filled += 1
                            except Exception as e:
                                print(f"Failed to fill field {current_key}: {e}")

        await fill_nested_data(form_data)
        return fields_filled

    # Rest of the class implementation remains the same...

    async def submit_llc_registration(self, json_data: Dict):
        """Main registration process with enhanced error handling and validation."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                # Initialize state and extract data
                state_info = json_data.get('data', {}).get('State', {})
                form_data = json_data.get('data', {}).get('Payload', {})

                # Navigate to state website
                await page.goto(state_info['stateUrl'])
                await page.wait_for_load_state("networkidle")

                # Handle login if credentials provided
                if state_info.get('filingWebsiteUsername'):
                    credentials = {
                        'username': state_info['filingWebsiteUsername'],
                        'password': state_info['filingWebsitePassword']
                    }
                    if not await self.handle_login(page, credentials):
                        raise Exception("Login failed")

                # Find and click registration button
                reg_elements = await self.detect_elements(page, "Identify LLC registration start button or link")
                if reg_button := await self.verify_element(page, reg_elements.get('register', {})):
                    await reg_button.click()
                    await page.wait_for_load_state("networkidle")

                # Fill form data
                fields_filled = await self.fill_form_data(page, form_data)
                print(f"Filled {fields_filled} form fields")

                # Submit form
                submit_elements = await self.detect_elements(page, "Identify form submit button")
                if submit_button := await self.verify_element(page, submit_elements.get('submit', {})):
                    await submit_button.click()
                    await page.wait_for_load_state("networkidle")

                    # Verify submission
                    try:
                        success = await page.wait_for_selector(
                            '[class*="success"], [class*="confirmation"]',
                            timeout=5000
                        )
                        if success:
                            print("Registration submitted successfully!")
                            return True
                    except:
                        print("Could not verify submission success")

                return False

            except Exception as e:
                print(f"Registration failed: {e}")
                return False
            finally:
                await browser.close()


async def main():
    api_key = API_KEY

    # Test data
    json_data = {
        "data": {
            "State": {
                "stateUrl": "https://www.sosnc.gov/",
                "filingWebsiteUsername": "redberyl",
                "filingWebsitePassword": "yD7?ddG0!$09"
            },
            "Payload": {
                "Entity_Formation": {
                    "Name": {
                        "CD_LLC_Name": "redberyl llc"
                    },
                    "Principal_Address": {
                        "PA_Address_Line_1": "123 Main Street",
                        "PA_City": "Solapur",
                        "PA_Zip_Code": "11557",
                        "PA_State": "AL"
                    }
                }
            }
        }
    }

    agent = LLCRegistrationAgent(api_key)
    await agent.submit_llc_registration(json_data)


if __name__ == "__main__":
    asyncio.run(main())