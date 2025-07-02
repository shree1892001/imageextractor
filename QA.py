import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from playwright.async_api import async_playwright, Page, ElementHandle
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from Common.constants import *


class LLCRegistrationAgent:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.field_patterns = {
            'login': [
                'login', 'log in', 'sign in', 'signin', 'connect', 'access', 'enter',
                'iniciar sesión', 'connecter', 'aanmelden', '登录', 'ログイン'
            ],
            'username': [
                'username', 'user name', 'email', 'phone', 'mobile', 'account',
                'usuario', 'utilisateur', 'gebruiker', '用户名', 'ユーザー'
            ],
            'password': [
                'password', 'pass word', 'passcode', 'pin', 'contraseña',
                'mot de passe', 'wachtwoord'
            ],
            'llc_name': [
                'company name', 'business name', 'llc name', 'entity name',
                'organization name'
            ],
            'registered_agent': [
                'registered agent', 'statutory agent', 'agent for service',
                'resident agent'
            ]
        }

        self.llm = GeminiModel(model, api_key=api_key)
        self.system_prompt = '''
        1. Automatically close any popups or notifications on the website.
        2. Automatically resolve the cloudflare captcha
        3. First, identify any initial login-related triggers:
           - "Sign In" or "Login" buttons/links that open login forms
           - Submit the sign in or login button to perform the login
           - Menu items or navigation elements that lead to login
           - Modal triggers or popups for login

        You are an advanced AI agent responsible for automating LLC registration form submissions across different state websites. Your task is to dynamically detect form fields, input the required data accurately, handle pop-ups or alerts, and ensure successful form submission.

        ### Task Execution Steps

        #### 1. Navigate to Registration Page
        - Open the stateUrl provided in jsonData.data.State.stateUrl
        - Wait for page to load completely
        - Identify and click appropriate link/button to start business filing
        - Select LLC entity type and proceed to form

        #### 2. Identify and Fill Required Fields
        - Dynamically detect all required fields on form
        - Fill values from payload
        - Ignore non-mandatory fields unless required for submission

        #### 3. LLC Name and Designator
        - Extract LLC name from payload.Name.CD_LLC_Name
        - Use payload.Name.CD_Alternate_Name for alternate name if required
        - Identify and select appropriate business designator
        - Enter LLC name and ensure compliance with form requirements

        #### 4. Registered Agent Information
        - Enter payload.Registered_Agent.RA_Email_Address for email fields
        - Handle business declarations (tobacco questions, management type)

        #### 5. Principal Office Address
        - Fill address fields with:
          - Street: payload.Principal_Address.PA_Address_Line_1
          - City: payload.Principal_Address.PA_City
          - State: payload.Principal_Address.PA_State
          - ZIP: payload.Principal_Address.PA_Zip_Code

        #### 6. Organizer Information
        - Enter payload.Organizer_Information.Org_Name if required

        #### 7. Registered Agent Details
        - Select individual/business entity type
        - Split and enter name from payload.Registered_Agent.RA_Name
        - Fill address with:
          - Street: payload.Registered_Agent.Address.RA_Address_Line_1
          - City: payload.Registered_Agent.Address.RA_City
          - ZIP: payload.Registered_Agent.Address.RA_Zip_Code

        #### 8. Registered Agent Signature
        - Input registered agent name for signature if required

        #### 9. Finalization and Submission
        - Check agreement/confirmation boxes
        - Click final submission button

        #### 10. Handle Pop-ups and Alerts
        - Detect and handle pop-ups, alerts, confirmation dialogs
        - Acknowledge and dismiss alerts before proceeding

        #### 11. Response and Error Handling
        - Return "Form filled successfully" on completion
        - Return "Form submission failed: <error>" on failure
        - Capture and report missing/error fields
        '''

        self.agent = Agent(model=self.llm, system_prompt=self.system_prompt, retries=3)

    async def handle_popups(self, page: Page):
        """Handle common popup types and notifications"""
        popup_selectors = [
            '[class*="popup"]', '[class*="modal"]', '[class*="dialog"]',
            '[class*="cookie"]', '[class*="notification"]'
        ]

        close_button_selectors = [
            'button:has-text("Close")', 'button:has-text("Accept")',
            '[class*="close"]', '[class*="dismiss"]', '×', 'x'
        ]

        for popup in popup_selectors:
            try:
                popup_element = await page.query_selector(popup)
                if popup_element and await popup_element.is_visible():
                    for close_button in close_button_selectors:
                        try:
                            close_el = await popup_element.query_selector(close_button)
                            if close_el:
                                await close_el.click()
                                await page.wait_for_timeout(1000)
                        except Exception:
                            continue
            except Exception:
                continue

    async def handle_cloudflare(self, page: Page):
        """Handle Cloudflare verification"""
        try:
            cloudflare_selectors = [
                '#challenge-form',
                '[class*="cloudflare"]',
                'iframe[src*="cloudflare"]'
            ]

            for selector in cloudflare_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(5000)  # Wait for verification
                except Exception:
                    continue

        except Exception as e:
            print(f"Error handling Cloudflare: {str(e)}")

    async def handle_login(self, page: Page, state_data: Dict) -> bool:
        """Handle website login process"""
        try:
            # Look for login triggers
            login_triggers = [
                'text="Login"', 'text="Sign In"',
                'button:has-text("Login")', 'button:has-text("Sign In")'
            ]

            for trigger in login_triggers:
                try:
                    element = await page.wait_for_selector(trigger, timeout=5000)
                    if element and await element.is_visible():
                        await element.click()
                        await page.wait_for_load_state("networkidle")
                        break
                except Exception:
                    continue

            # Fill username/email
            username_field = await self.smart_field_detection(page, 'username')
            if username_field:
                await username_field.fill(state_data['filingWebsiteUsername'])

            # Fill password
            password_field = await self.smart_field_detection(page, 'password')
            if password_field:
                await password_field.fill(state_data['filingWebsitePassword'])

            # Submit login
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button[type="button"]',
                'button:has-text("Login")',
                'button:has-text("Sign In")'
            ]

            for selector in submit_selectors:
                try:
                    submit_button = await page.wait_for_selector(selector, timeout=5000)
                    if submit_button and await submit_button.is_visible():
                        await submit_button.click()
                        await page.wait_for_load_state("networkidle")
                        break
                except Exception:
                    continue

            # Verify login success
            success_indicators = [
                '[class*="dashboard"]',
                '[class*="account"]',
                '[class*="logged-in"]'
            ]

            for indicator in success_indicators:
                try:
                    await page.wait_for_selector(indicator, timeout=5000)
                    return True
                except Exception:
                    continue

            return False

        except Exception as e:
            print(f"Login error: {str(e)}")
            return False

    async def smart_field_detection(self, page: Page, field_type: str) -> ElementHandle:
        """Intelligently detect form fields based on context and attributes"""
        selectors = []

        if field_type in self.field_patterns:
            patterns = self.field_patterns[field_type]
            for pattern in patterns:
                selectors.extend([
                    f'input[name*="{pattern}" i]',
                    f'input[id*="{pattern}" i]',
                    f'input[placeholder*="{pattern}" i]',
                    f'label:has-text("{pattern}")',
                    f'div:has-text("{pattern}") input'
                ])

        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if element and await element.is_visible():
                    return element
            except Exception:
                continue

        return None

    async def fill_llc_data(self, page: Page, json_data: Dict):
        """Fill LLC registration form fields"""
        try:
            # Extract LLC information
            payload = json_data['data']['Payload']['Entity_Formation']

            # Fill LLC Name
            llc_name_field = await self.smart_field_detection(page, 'llc_name')
            if llc_name_field:
                await llc_name_field.fill(payload['Name']['CD_LLC_Name'])

            # Principal Address
            pa_data = payload['Principal_Address']
            address_fields = {
                'address': pa_data['PA_Address_Line_1'],
                'address2': pa_data.get('PA_Address_Line_2', ''),
                'city': pa_data['PA_City'],
                'state': pa_data['PA_State'],
                'zip': pa_data['PA_Zip_Code']
            }

            for field_type, value in address_fields.items():
                field = await self.smart_field_detection(page, field_type)
                if field:
                    await field.fill(value)

            # Registered Agent
            agent_data = payload['Registered_Agent']
            ra_fields = {
                'name': agent_data['RA_Name'],
                'email': agent_data['RA_Email_Address'],
                'phone': agent_data['RA_Contact_No'],
                'address': agent_data['Address']['RA_Address_Line_1'],
                'city': agent_data['Address']['RA_City'],
                'state': agent_data['Address']['RA_State'],
                'zip': agent_data['Address']['RA_Zip_Code']
            }

            for field_type, value in ra_fields.items():
                field = await self.smart_field_detection(page, field_type)
                if field:
                    await field.fill(value)

            # Organizer Information
            if 'Organizer_Information' in payload:
                org_data = payload['Organizer_Information']['Organizer_Details']
                org_fields = {
                    'name': org_data['Org_Name'],
                    'email': org_data['Org_Email_Address'],
                    'phone': org_data['Org_Contact_No']
                }

                for field_type, value in org_fields.items():
                    field = await self.smart_field_detection(page, field_type)
                    if field:
                        await field.fill(value)

            # Handle any remaining fields
            await self.detect_and_fill_remaining_fields(page, payload)

        except Exception as e:
            print(f"Error filling LLC data: {str(e)}")

    async def detect_and_fill_remaining_fields(self, page: Page, payload: Dict):
        """Detect and fill any remaining form fields based on context"""
        all_inputs = await page.query_selector_all('input:not([type="hidden"]), select, textarea')

        for input_element in all_inputs:
            try:
                field_info = await self.extract_field_info(input_element)
                if not field_info.get('value'):  # Only fill empty fields
                    value = await self.find_matching_value(field_info, payload)
                    if value:
                        await input_element.fill(str(value))
            except Exception as e:
                print(f"Error filling field: {str(e)}")
                continue

    async def extract_field_info(self, element: ElementHandle) -> Dict:
        """Extract detailed information about a form field"""
        info = await element.evaluate('''
            element => {
                const getLabel = (el) => {
                    let label = '';
                    // Check for label element
                    if (el.labels && el.labels.length) {
                        label = el.labels[0].textContent;
                    }
                    // Check aria-label
                    if (!label && el.getAttribute('aria-label')) {
                        label = el.getAttribute('aria-label');
                    }
                    // Check placeholder
                    if (!label && el.placeholder) {
                        label = el.placeholder;
                    }
                    // Check preceding text
                    if (!label) {
                        let previous = el.previousSibling;
                        while (previous) {
                            if (previous.nodeType === 3) {  // Text node
                                label = previous.textContent.trim();
                                if (label) break;
                            }
                            previous = previous.previousSibling;
                        }
                    }
                    return label.trim();
                };

                return {
                    type: element.type || element.tagName.toLowerCase(),
                    name: element.name,
                    id: element.id,
                    label: getLabel(element),
                    value: element.value,
                    required: element.required,
                    pattern: element.pattern,
                    maxLength: element.maxLength,
                    minLength: element.minLength,
                    placeholder: element.placeholder
                };
            }
        ''')
        return info

    async def find_matching_value(self, field_info: Dict, payload: Dict) -> Optional[str]:
        """Find matching value from payload based on field context"""
        prompt = f"""
        Analyze this form field and find the best matching value from the payload:

        Field Info:
        {json.dumps(field_info, indent=2)}

        Payload:
        {json.dumps(payload, indent=2)}

        Return the most appropriate value or null if no match found.
        """

        result = await self.agent.run(prompt)
        try:
            value = json.loads(result.data)
            return value if value else None
        except:
            return None

    async def submit_form(self, page: Page) -> bool:
            """Submit the form and verify success"""
            try:
                # Handle any final confirmations
                await self.handle_confirmations(page)

                # Check for missing required fields
                missing_fields = await self.check_required_fields(page)
                if missing_fields:
                    print(f"Missing required fields: {', '.join(missing_fields)}")
                    return False

                # Find and click submit button
                submit_selectors = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Submit")',
                    'button:has-text("File")',
                    'button:has-text("Continue")'
                ]

                for selector in submit_selectors:
                    try:
                        submit_button = await page.wait_for_selector(selector, timeout=5000)
                        if submit_button and await submit_button.is_visible():
                            await submit_button.click()
                            await page.wait_for_load_state("networkidle")
                            break
                    except Exception:
                        continue

                # Verify submission success
                success_indicators = [
                    '[class*="success"]',
                    '[class*="confirmation"]',
                    'text="Thank you"',
                    'text="Successfully"'
                ]

                for indicator in success_indicators:
                    try:
                        await page.wait_for_selector(indicator, timeout=10000)

                        # Try to capture confirmation number if available
                        confirmation_selectors = [
                            '[class*="confirmation-number"]',
                            '[class*="reference"]',
                            'text=/Confirmation: \d+/',
                            'text=/Reference: \d+/'
                        ]

                        for conf_selector in confirmation_selectors:
                            try:
                                conf_element = await page.wait_for_selector(conf_selector, timeout=2000)
                                if conf_element:
                                    confirmation = await conf_element.text_content()
                                    print(f"Confirmation: {confirmation}")
                                    break
                            except Exception:
                                continue

                        return True
                    except Exception:
                        continue

                return False

            except Exception as e:
                print(f"Error submitting form: {str(e)}")
                return False

    async def submit_llc_registration(self, json_data: Dict):
            """Main method to handle LLC registration process"""
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context()
                page = await context.new_page()

                try:
                    # Navigate to state website
                    await page.goto(json_data['data']['State']['stateUrl'])
                    await page.wait_for_load_state("networkidle")

                    # Handle initial page setup
                    await self.handle_cloudflare(page)
                    await self.handle_popups(page)

                    # Handle login if required
                    if json_data['data']['State'].get('filingWebsiteUsername'):
                        login_success = await self.handle_login(page, json_data['data']['State'])
                        if not login_success:
                            print("Login failed")
                            return "Login failed"

                    # Navigate to LLC registration form
                    await self.navigate_to_registration(page)

                    # Fill form data
                    await self.fill_llc_data(page, json_data)

                    # Submit form
                    success = await self.submit_form(page)
                    if success:
                        return "Form filled successfully"
                    else:
                        return "Form submission failed: Unable to confirm success"

                except Exception as e:
                    error_msg = f"Error during LLC registration: {str(e)}"
                    print(error_msg)
                    return error_msg
                finally:
                    await browser.close()
    async def navigate_to_registration(self, page: Page):
            """Navigate to LLC registration form"""
            registration_links = [
                'text="Register a Business"',
                'text="Start a Business"',
                'text="Form an LLC"',
                'text="Create LLC"',
                'text="File a Business"'
            ]

            for link in registration_links:
                try:
                    element = await page.wait_for_selector(link, timeout=5000)
                    if element:
                        await element.click()
                        await page.wait_for_load_state("networkidle")
                        break
                except Exception:
                    continue

async def main():
        api_key = API_KEY

        json_data = {
            "data": {
                "EntityType": {
                    "id": 1,
                    "entityShortName": "LLC",
                    "entityFullDesc": "Limited Liability Company",
                    "onlineFormFilingFlag": False
                },
                "State": {
                    "id": 33,
                    "stateShortName": "NC",
                    "stateFullDesc": "North Carolina",
                    "stateUrl": "https://www.sosnc.gov/",
                    "filingWebsiteUsername": "redberyl",
                    "filingWebsitePassword": "yD7?ddG0!$09",
                    "strapiDisplayName": "North-Carolina",
                    "countryMaster": {
                        "id": 3,
                        "countryShortName": "US",
                        "countryFullDesc": "United States"
                    }
                },
                "County": {
                    "id": 2006,
                    "countyCode": "Alleghany",
                    "countyName": "Alleghany",
                    "stateId": {
                        "id": 33,
                        "stateShortName": "NC",
                        "stateFullDesc": "North Carolina",
                        "stateUrl": "https://www.sosnc.gov/",
                        "filingWebsiteUsername": "redberyl",
                        "filingWebsitePassword": "yD7?ddG0!$09",
                        "strapiDisplayName": "North-Carolina",
                        "countryMaster": {
                            "id": 3,
                            "countryShortName": "US",
                            "countryFullDesc": "United States"
                        }
                    }
                },
                "Payload": {
                    "Entity_Formation": {
                        "Name": {
                            "CD_LLC_Name": "redberyl llc",
                            "CD_Alternate_LLC_Name": "redberyl llc"
                        },
                        "Principal_Address": {
                            "PA_Address_Line_1": "123 Main Street",
                            "PA_Address_Line_2": "",
                            "PA_City": "Solapur",
                            "PA_Zip_Code": "11557",
                            "PA_State": "AL"
                        },
                        "Registered_Agent": {
                            "RA_Name": "Interstate Agent Services LLC",
                            "RA_Email_Address": "agentservice@vstatefilings.com",
                            "RA_Contact_No": "(718) 569-2703",
                            "Address": {
                                "RA_Address_Line_1": "6047 Tyvola Glen Circle, Suite 100",
                                "RA_Address_Line_2": "",
                                "RA_City": "Charlotte",
                                "RA_Zip_Code": "28217",
                                "RA_State": "NC"
                            }
                        },
                        "Billing_Information": {
                            "BI_Name": "Johson Charles",
                            "BI_Email_Address": "johson.charles@redberyktech.com",
                            "BI_Contact_No": "(555) 783-9499",
                            "BI_Address_Line_1": "123 Main Street",
                            "BI_Address_Line_2": "",
                            "BI_City": "Albany",
                            "BI_Zip_Code": "68342",
                            "BI_State": "AL"
                        },
                        "Mailing_Information": {
                            "MI_Name": "Johson Charles",
                            "MI_Email_Address": "johson.charles@redberyktech.com",
                            "MI_Contact_No": "(555) 783-9499",
                            "MI_Address_Line_1": "123 Main Street",
                            "MI_Address_Line_2": "",
                            "MI_City": "Albany",
                            "MI_Zip_Code": "68342",
                            "MI_State": "AL"
                        },
                        "Organizer_Information": {
                            "Organizer_Details": {
                                "Org_Name": "Johson Charles",
                                "Org_Email_Address": "johson.charles@redberyktech.com",
                                "Org_Contact_No": "(555) 783-9499"
                            },
                            "Address": {
                                "Org_Address_Line_1": "123 Main Street",
                                "Org_Address_Line_2": "",
                                "Org_City": "Albany",
                                "Org_Zip_Code": "68342",
                                "Org_State": "AL"
                            }
                        }
                    }
                }
            }

        }
        agent = LLCRegistrationAgent(api_key)
        await agent.submit_llc_registration(json_data)

if __name__ == "__main__":
        asyncio.run(main())