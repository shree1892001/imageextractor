API_KEY= "AIzaSyDJvOAxCnVXB23FqPUadFwQIZay1ESNEIk" or "AIzaSyDcBP8bIBztxmXJh1AL5PxkaaEFfnzSBgc"
OPENAI_API = "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"
DB_NAME = "test4"
DB_USER = "postgres"
DB_PASS  = "postgres"
DB_PORT= 5432
DB_SERVER_NAME="local_shreyas"
DB_HOST = "127.0.0.1"
DB_NAME = "test5"
CLASSIFICATION_PROMPT = """
Analyze this document image and identify the type of document by extracting only the following specific keywords if they are present: "license," "Pancard," or "aadharcard." , "passport," . Return the result in the following JSON format:        {
            "document_type": "The type of document (e.g., 'Pancard', 'License', 'AadhaarCard', 'Passport' ,'SSN' ,'passport' etc.)",
            "confidence_score": "A score between 0 and 1 indicating confidence in classification",
            "document_features": ["List of key features identified that helped in classification"]
        }
        Be specific with the document type and ensure that the document is valid  and only classify if confident.
        """


AADHAR_CARD_EXTRACTION = """
        Extract the following fields from the Aadhaar card in JSON format:
        {
            "document_type": "AadhaarCard",
            "data": {
                "aadhaar_number": "",
                "name": "",
                "gender": "",
                "date_of_birth": "",
                "address": "",
                "postal_code": ""
            }
        }
        Ensure Aadhaar number is properly formatted and dates are in YYYY-MM-DD format.
        """
PAN_CARD_EXTRACTION = """
        Extract the following fields from the PAN card in JSON format:
        {
            "document_type": "PAN_Card",
            "data": {
                "pan_number": "",
                "name": "",
                "fathers_name": "",
                "date_of_birth": "",
                "issue_date": ""
            }
        }
        Ensure PAN number is in correct format (AAAPL1234C) and dates are in YYYY-MM-DD format.
        """
LICENSE_EXTRACTION = """
        Extract the following fields from the driving license in JSON format:
        {
            "document_type": "License",
            "data": {
                "license_number": "",
                "name": "",
                "date_of_birth": "",
                "address": "",
                "valid_from": "",
                "valid_until": "",
                "vehicle_categories": [],
                "issuing_authority": ""
            }
        }
        Ensure all dates are in YYYY-MM-DD format and text fields are properly cased.
        """

PASSPORT_EXTRACTION = """ 

 Extract the following fields from the Passport in JSON format:
{
    "document_type": "Passport",
    "data": {
        "passport_number": "",
        "surname": "",
        "given_names": "",
        "nationality": "",
        "date_of_birth": "",
        "place_of_birth": "",
        "gender": "",
        "date_of_issue": "",
        "date_of_expiry": "",
        "place_of_issue": "",
        "type": "",
        "country_code": ""
    }
}
- Passport number format:
    * For US passports: 9 alphanumeric characters (e.g., 123456789 or C12345678).
    * For other countries: May start with an uppercase letter, followed by 7–9 digits.
- Dates should be in ISO format (YYYY-MM-DD).
- Country code must be a valid 3-letter ISO country code (e.g., IND for India, USA for United States).
- Gender should be one of: M (Male), F (Female), or X (Unspecified).
- Type must be one of the following: 
    * P (Personal)
    * D (Diplomatic)
    * S (Service)
Ensure extracted data adheres to these standards.

"""

jsonData = {

       "jsonData":{
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
AUTOMATION_TASK= f"""
      ### **Advanced AI Agent for Automated LLC Registration** 
      
      -Confirm image is clickable
      - Check visibility in viewport
      - Verify enabled state
      - Match to correct 
      - Try direct image click
       - Click parent container if needed
        - Use element index if provided
       - Click associated label if necessary
       -Click the button
      In case of 400 error reload the page and continue the automation from the point left  
      -Interact with the elements even though they are images not proper input fields.
      
      --Select the button with text "Start Filing" or Begin Filing or Start Register Business  if found even though its an image click the button even if its an image.
      You are an advanced AI agent responsible for automating LLC registration form submissions across different state websites. Your task is to dynamically detect form fields, input the required data accurately, handle pop-ups or alerts, and ensure successful form submission. The AI should adapt to varying form structures and selectors without relying on predefined element locators.  
       If their are questions asked on the site like Has this entity been created in another state or country? or similar then select No from the dropdown 
       -Properly select all the fields and ensure that the fields are populated accurately
       - Select the LLC entity type: `${jsonData["jsonData"]["EntityType"]["entityShortName"]}` or .`${jsonData["jsonData"]["EntityType"]["entityFullDesc"]}` from the dropdown or from any relevent field. 

       
      ### **Task Execution Steps**  

      #### **1. Navigate to the Registration Page**  
    - Go to the url `${jsonData["jsonData"]["State"]["stateUrl"]}` url.  
    - Wait for the page to load completely.  

    #### **2. Handle Pop-ups and Initial UI Elements**  
    - Automatically close any pop-ups, notifications, or modals.  
    - Detect and handle Cloudflare captcha if present.  
    - Identify any initial login-related triggers:  
         - "Sign In" or "Login" buttons/links that open login forms  
    - Menu items or navigation elements that lead to login  
    - Modal triggers or popups for login  

#### **3. Perform Login (If Required)**  
- If a login form appears, identify:  
  - Username/email input field  
  - Password input field  
  - Login/Submit button  
- Enter credentials from the JSON:  
  - Username: `${jsonData["jsonData"]["State"]["filingWebsiteUsername"]}`  
  - Password: `${jsonData["jsonData"]["State"]["filingWebsitePassword"]}`  
- Click the login button and wait for authentication to complete.  

#### **4. Start LLC Registration Process**  
- Identify and click the appropriate link or button to start a new business  filing or Register  New Business button .
 
 
- Select the LLC entity type: `${jsonData["jsonData"]["EntityType"]["entityShortName"]}` or .`${jsonData["jsonData"]["EntityType"]["entityFullDesc"]}` from the dropdown or from any relevent field. 
 - if the site ask for the options file online or upload the pdf select or click the file online button or select it from dropdown or from checkbox 
 -If a button has text like "Start Filing", "Begin Filing", or "Start Register Business", click it—whether it's a standard button or an image.
 -If we need to save the name then click the save the name button or proceed next button.
- Proceed to the form.  

#### **5. Identify and Fill Required Fields**  
- Dynamically detect all required fields on the form and fill in the values from `${jsonData["jsonData"]["Payload"] }` make sure to flatten it at is dynamic json.  
- Ignore non-mandatory fields unless explicitly required for submission.  

#### **6. LLC Name and Designator**  
- Extract the LLC name from `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Name"]["CD_LLC_Name"]}`.  
- If  LLC a name does not work then replace the LLC name with the Alternate llc name  , use `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Name"]["CD_Alternate_LLC_Name"]}`.  
- Identify and select the appropriate business designator.  
- Enter the LLC name and ensure compliance with form requirements.  

#### **7. Registered Agent Information**  
- If an email field is detected, enter `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["RA_Email_Address"]}`. 

- Identify and respond to any required business declarations (e.g., tobacco-related questions, management type).  

#### **8. Principal Office Address** (If Required)  
- Detect address fields and input the values accordingly:  
  - Street Address: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_Address_Line_1"]}`.  
  - City: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_City"]}`.  
  - State: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_State"]}`.  
  - ZIP Code: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_Zip_Code"]}`.  

#### **9. Organizer Information** (If Required)  
- If the form includes an organizer section, enter `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Organizer_Information"]["Organizer_Details"]["Org_Name"]}`.  

#### **10. Registered Agent Details**  
-Enter the Registered Agent details in its respective fields only by identifying the label for Registered Agent
- Detect and select if the registered agent is an individual or business entity.  
- If required, extract and split the registered agent’s full name   "from `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["RA_Name"]}`, then input:  
  - First Name  
  - Last Name  
  -If for example the name of the registered agent is Interstate Agent Services LLC then the  First Name would be "Interstate" and the Last Name would be "Agent Services LLC"
- If an address field is present, enter:  
  - Street Address/ Address Line_1 `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_Address_Line_1"]}`.  
  - City: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_City"]}`.  
  - ZIP Code or Zip Code or similar field: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_Zip_Code"]}`.  
  - IF  in the address their is requirement of County , select `${jsonData['jsonData']['County']['countyName']} either from dropdown or enter the value in it 

#### **11. Registered Agent Signature (If Required)**  
- If a signature field exists, input the registered agent’s first and last name.  

#### **12. Finalization and Submission**  
- Identify and check any agreement or confirmation checkboxes.  
- Click the final submission button to complete the filing.  

#### **13. Handling Pop-ups, Alerts, and Dialogs**  
- Detect and handle any pop-ups, alerts, or confirmation dialogs.  
- If an alert appears, acknowledge and dismiss it before proceeding.  

#### **14. Response and Error Handling**  
- Return `"Form filled successfully"` upon successful completion.  
- If an error occurs, log it and return `"Form submission failed: <error message>"`.  
- If required fields are missing or contain errors, capture the issue and provide feedback on what needs to be corrected.  

### **AI Agent Execution Guidelines**  
- Dynamically detect and interact with form elements without relying on predefined selectors.  
- Adapt to different form structures and ignore unnecessary fields.  
- Handle UI changes and errors efficiently, ensuring smooth automation.  
- Maintain accuracy and compliance while minimizing user intervention.  

    
"""



AUTOMATION_TASK1= f"""
# Advanced AI Agent for Automated LLC Registration

## Core Rules and Error Recovery
1. **Never Stop on Failure**
   - Attempt 3 retries for each action
   - Wait 2-3 seconds between retries
   - Log errors but continue execution
   - Save progress after each successful step

2. **Page Load Handling**
   - Allow 30 seconds for page load
   - Refresh page if load fails
   - Handle timeouts with automatic retry
   - Check for network connectivity

## Data Fields and Mapping

### 1. Initial Navigation
- Go to State URL: `${jsonData["jsonData"]["State"]["stateUrl"]}`
- If login required:
  - Username: `${jsonData["jsonData"]["State"]["filingWebsiteUsername"]}`
  - Password: `${jsonData["jsonData"]["State"]["filingWebsitePassword"]}`

### 2. Start Filing Process
- Look for and click buttons with text:
  - "Start Filing"
  - "Begin Filing"
  - "Start Register Business"
  - "Register New Business"
- If presented with options:
  - Select "file online" over "upload pdf"
  - Choose "Start Filing" or similar button

### 3. Entity Selection
- Select LLC entity type using either:
  - Short name: `${jsonData["jsonData"]["EntityType"]["entityShortName"]}`
  - Full name: `${jsonData["jsonData"]["EntityType"]["entityFullDesc"]}`

### 4. LLC Name Entry
- Primary search patterns: "Company Name", "Business Name", "LLC Name"
- Values to try in order:
  1. Primary: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Name"]["CD_LLC_Name"]}`
  2. Alternate: `${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Name"]["CD_Alternate_LLC_Name"]}`

### 5. Registered Agent Information
Field patterns and values:
```
Name Fields ("Registered Agent", "Agent Name", "Statutory Agent"):
- Full Name: ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["RA_Name"]}
  If split name required:
  - First Name: Take first word
  - Last Name: Take remaining words

Email ("Agent Email", "Email Address"):
- ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["RA_Email_Address"]}

Address Fields:
- Street ("Address Line 1", "Street Address"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_Address_Line_1"]}
- City ("City", "Town"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_City"]}
- State:
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_State"]}
- ZIP ("ZIP Code", "Postal Code"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Registered_Agent"]["Address"]["RA_Zip_Code"]}
```

### 6. Principal Office Address
Field patterns and values:
```
Look for: "Principal Office", "Business Address", "Main Address"

- Street ("Address Line 1", "Street Address"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_Address_Line_1"]}
- City ("City", "Town"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_City"]}
- State:
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_State"]}
- ZIP ("ZIP Code", "Postal Code"):
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Principal_Address"]["PA_Zip_Code"]}
```

### 7. Organizer Information
Field patterns and values:
```
Look for: "Organizer", "Organizer Information", "Organizer Details"

- Name:
  ${jsonData["jsonData"]["Payload"]["Entity_Formation"]["Organizer_Information"]["Organizer_Details"]["Org_Name"]}
```

### 8. County Information (if required)
```
Look for: "County", "County Name", "Business County"
Value: ${jsonData['jsonData']['County']['countyName']}
```

## Field Population Rules

1. **Field Detection**
   - Search for exact label matches first
   - Try partial matches if exact fails
   - Check placeholder text
   - Look for aria-labels

2. **Input Verification**
   - Verify each field after population
   - Check for validation errors
   - Try alternative input methods if needed
   - Wait for dynamic validation

3. **Special Cases**
   - If dropdown, search all options for match
   - If radio/checkbox, find closest match
   - If multi-step form, track progress
   - Handle dynamic form updates

## Error Recovery Actions

1. **For Each Field**:
   ```
   - Try direct input
   - Wait 2 seconds if fails
   - Try clicking field first
   - Try force focus
   - Verify input accepted
   ```

2. **For Button Clicks**:
   ```
   - Try direct click
   - Try JavaScript click
   - Try parent element
   - Try keyboard Enter
   ```

3. **For Failed Steps**:
   ```
   - Log error details
   - Try alternative approach
   - Save partial progress
   - Continue to next step
   ```

## Success Criteria
- All required fields populated
- No validation errors
- Form submission confirmed
- Confirmation number received

## Progress Tracking
- Save state after each step
- Log completed actions
- Track remaining steps
- Maintain error log

Remember:
1. Never stop on non-critical errors
2. Try all alternative values if primary fails
3. Keep detailed logs of actions and errors
4. Save progress regularly
5. Handle alerts and popups automatically

If their are questions asked on the site like "Has this entity been created in another state or country?" or similar then select "No" from the dropdown.
"""





SSN_EXTRACTION = "Extract the following fields from the SSN document: ssn, name, date_of_birth, address."
