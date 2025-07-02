

import fitz  # PyMuPDF
import google.generativeai as genai
import json
import re
from pdfrw import PdfReader, PdfWriter

# Step 1: Extract Form Fields
def extract_form_fields(pdf_path):
    """Extracts all form field names from a fillable PDF."""
    pdf = PdfReader(pdf_path)
    field_names = []

    for page in pdf.pages:
        annotations = page.Annots or []
        for annotation in annotations:
            key = annotation.T and annotation.T[1:-1]
            if key:
                field_names.append(key)

    return field_names

# Step 2: Extract PDF Text
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

# Step 3: Extract Nearby Text Context
def extract_field_context(pdf_path, form_fields):
    """Extracts context (nearby text) for each form field."""
    doc = fitz.open(pdf_path)
    field_context = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        for field in form_fields:
            match = re.search(rf"(.{{0,50}}{field}.{{0,50}})", text, re.IGNORECASE)
            if match:
                field_context[field] = match.group(0)

    return field_context

# Step 4: AI Mapping
def get_mapped_fields(pdf_text, form_fields, field_context, payload):
    """Uses AI to correctly match form fields with values."""
    genai.configure(api_key="AIzaSyBdtg1qO3lxz5axHFxSIlqau2NumePDF30")
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Given the extracted text from a PDF:
    ---
    {pdf_text}
    ---

    The following form fields were extracted:
    ---
    {form_fields}
    ---

    Nearby text context:
    ---
    {json.dumps(field_context, indent=4)}
    ---

    User's provided payload:
    ---
    {json.dumps(payload, indent=4)}
    ---

    Match **only** the extracted form fields to their correct values.
    - Do **NOT** create new labels.
    - Prioritize payload values if available.
    - If no match is found, return **None**.

    **Return a JSON object where keys are exact field names from the form.**
    """

    response = model.generate_content(prompt)
    json_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""
    json_text = json_text.replace("```json", "").replace("```", "").strip()

    return json.loads(json_text)

# Step 5: Fill PDF Form
def fill_pdf_form(pdf_path, output_pdf_path, mapped_fields):
    """Fills the PDF form with AI-mapped values."""
    pdf = PdfReader(pdf_path)

    for page in pdf.pages:
        annotations = page.Annots or []
        for annotation in annotations:
            key = annotation.T and annotation.T[1:-1]
            if key in mapped_fields and mapped_fields[key]:  # Ignore empty fields
                annotation.V = f"({mapped_fields[key]})"

    PdfWriter(output_pdf_path, trailer=pdf).write()
    print(f"✅ Updated PDF saved at: {output_pdf_path}")

# Run the Automation
pdf_path = "D:\\demo\\Services\\Maine.pdf"
output_pdf_path = "D:\\demo\\Services\\fill_smart12.pdf"

payload = {
    "Entity_Formation": {
      "Name": {
        "CD_LLC_Name": "Infosys Corp."
      },
      "Principal_Address": {
        "PA_Address_Line_1": "Iowa State University Main Campus",
        "PA_Address_Line_2": "",
        "PA_City": "Ames",
        "PA_Zip_Code": "50011",
        "PA_State": "IA"
      },
      "Registered_Agent": {
        "RA_Name": "Interstate Agent Services LLC",
        "RA_Email_Address": "agentservice@vstatefilings.com",
        "RA_Contact_No": "(718) 569-2703",
        "Address": {
          "RA_Address_Line_1": "3000 Atrium Way, Ste 265",
          "RA_Address_Line_2": "",
          "RA_City": "Mt. Laurel",
          "RA_Zip_Code": "08054",
          "RA_State": "NJ"
        }
      },
      "Purpose": {
        "CD_Business_Purpose_Details": "Any Lawful Purpose",
        "CD_Please_specify": ""
      },
      "Organizer_Information": {
        "Organizer_Details": {
          "Org_Name": "Riyan Parag",
          "Org_Email_Address": "riyan.parag@gmail.com",
          "Org_Contact_No": "(879) 877-8784"
        },
        "Address": {
          "Org_Address_Line_1": "Disneyland Resort",
          "Org_Address_Line_2": "",
          "Org_City": "Anaheim",
          "Org_Zip_Code": "54564",
          "Org_State": "CA"
        }
      },
      "Member_Or_Manager_Details": [
        {
          "__temp_key__": 0,
          "Mom_Member_Or_Manager": "Member",
          "member_or_manger_details": {
            "createdBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
            "creationDate": "2025-02-19T06:33:04.709+0000",
            "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
            "lastModifiedDate": "2025-02-19T06:33:04.709+0000",
            "id": 470,
            "companyId": {
              "createdBy": "null",
              "creationDate": "2025-02-19T05:23:28.070+0000",
              "lastModifiedBy": "null",
              "lastModifiedDate": "2025-02-19T05:24:02.973+0000",
              "id": 225,
              "companyName": "PrimeStart",
              "companyEmail": "null",
              "phoneNo": "null",
              "websiteUrl": "null",
              "fax": "null",
              "accountManagerId": "null",
              "adminContactId": {
                "createdBy": "null",
                "creationDate": "2025-02-19T05:23:28.060+0000",
                "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
                "lastModifiedDate": "2025-02-19T06:14:51.061+0000",
                "id": 197,
                "salutation": "Mrs",
                "firstName": "Snehal",
                "lastName": "Hendre",
                "displayName": "Snehal Hendre (V-1200256)",
                "jobTitle": "Java Developer",
                "contactSourceId": "null",
                "recordTypeId": "null",
                "avatarImgPath": "null",
                "phone": "null",
                "fax": "null",
                "emailId": "vasin38197@codverts.com",
                "mobileNo": "(987) 545-6223",
                "statusId": 1,
                "userId": 1200256,
                "quickbooksId": 1134,
                "hubspotId": 100496920462,
                "vendorDetails": "null"
              },
              "primaryContactId": {
                "createdBy": "null",
                "creationDate": "2025-02-19T05:23:28.060+0000",
                "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
                "lastModifiedDate": "2025-02-19T06:14:51.061+0000",
                "id": 197,
                "salutation": "Mrs",
                "firstName": "Snehal",
                "lastName": "Hendre",
                "displayName": "Snehal Hendre (V-1200256)",
                "jobTitle": "Java Developer",
                "contactSourceId": "null",
                "recordTypeId": "null",
                "avatarImgPath": "null",
                "phone": "null",
                "fax": "null",
                "emailId": "vasin38197@codverts.com",
                "mobileNo": "(987) 545-6223",
                "statusId": 1,
                "userId": 1200256,
                "quickbooksId": 1134,
                "hubspotId": 100496920462,
                "vendorDetails": "null"
              },
              "industryId": "null",
              "organizerContactId": "null",
              "entityType": "null",
              "naicsCode": "null",
              "formationDate": "null",
              "dissolutionDate": "null",
              "boardCertifiedFlag": "null",
              "serviceStateId": "null",
              "domesticStateId": "null",
              "einNo": "null",
              "entityNo": "null",
              "taxYearEnd": "null",
              "registeredAgentName": "null",
              "registeredAgentAddress": "null",
              "registeredAgentEmail": "null",
              "registeredAgentContactNo": "null",
              "dbaName": "null",
              "fkaName": "null",
              "quickbooksId": "null",
              "hubspotId": 30140779578,
              "statusId": 4,
              "documentsJson": "null",
              "documentIds": []
            },
            "keyPersonnelTypeId": 20,
            "keyPersonnelName": "Mayuri Chavan",
            "keyPersonnelTitle": "",
            "dateOfTransfer": "null",
            "ownershipPercentage": "null",
            "contactNo": "+1 (456) 595-2266",
            "emailId": "mayuri@gmail.com",
            "addressMasterId": {
              "createdBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
              "creationDate": "2025-02-19T06:33:02.909+0000",
              "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
              "lastModifiedDate": "2025-02-19T06:33:02.909+0000",
              "id": 772,
              "addressLine1": "507 B Amanora Mall",
              "addressLine2": "null",
              "city": "Arizona",
              "stateId": 3,
              "postalCode": "11557",
              "countryId": 3,
              "contactDetails": "null",
              "companyDetails": {
                "createdBy": "null",
                "creationDate": "2025-02-19T05:23:28.070+0000",
                "lastModifiedBy": "null",
                "lastModifiedDate": "2025-02-19T05:24:02.973+0000",
                "id": 225,
                "companyName": "PrimeStart",
                "companyEmail": "null",
                "phoneNo": "null",
                "websiteUrl": "null",
                "fax": "null",
                "accountManagerId": "null",
                "adminContactId": {
                  "createdBy": "null",
                  "creationDate": "2025-02-19T05:23:28.060+0000",
                  "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
                  "lastModifiedDate": "2025-02-19T06:14:51.061+0000",
                  "id": 197,
                  "salutation": "Mrs",
                  "firstName": "Snehal",
                  "lastName": "Hendre",
                  "displayName": "Snehal Hendre (V-1200256)",
                  "jobTitle": "Java Developer",
                  "contactSourceId": "null",
                  "recordTypeId": "null",
                  "avatarImgPath": "null",
                  "phone": "null",
                  "fax": "null",
                  "emailId": "vasin38197@codverts.com",
                  "mobileNo": "(987) 545-6223",
                  "statusId": 1,
                  "userId": 1200256,
                  "quickbooksId": 1134,
                  "hubspotId": 100496920462,
                  "vendorDetails": "null"
                },
                "primaryContactId": {
                  "createdBy": "null",
                  "creationDate": "2025-02-19T05:23:28.060+0000",
                  "lastModifiedBy": "ENC:uE9Dx0VurIyAq1cz65l0M5eD+yvz+20Jnh4cDegKt5o=",
                  "lastModifiedDate": "2025-02-19T06:14:51.061+0000",
                  "id": 197,
                  "salutation": "Mrs",
                  "firstName": "Snehal",
                  "lastName": "Hendre",
                  "displayName": "Snehal Hendre (V-1200256)",
                  "jobTitle": "Java Developer",
                  "contactSourceId": "null",
                  "recordTypeId": "null",
                  "avatarImgPath": "null",
                  "phone": "null",
                  "fax": "null",
                  "emailId": "vasin38197@codverts.com",
                  "mobileNo": "(987) 545-6223",
                  "statusId": 1,
                  "userId": 1200256,
                  "quickbooksId": 1134,
                  "hubspotId": 100496920462,
                  "vendorDetails": "null"
                },
                "industryId": "null",
                "organizerContactId": "null",
                "entityType": "null",
                "naicsCode": "null",
                "formationDate": "null",
                "dissolutionDate": "null",
                "boardCertifiedFlag": "null",
                "serviceStateId": "null",
                "domesticStateId": "null",
                "einNo": "null",
                "entityNo": "null",
                "taxYearEnd": "null",
                "registeredAgentName": "null",
                "registeredAgentAddress": "null",
                "registeredAgentEmail": "null",
                "registeredAgentContactNo": "null",
                "dbaName": "null",
                "fkaName": "null",
                "quickbooksId": "null",
                "hubspotId": 30140779578,
                "statusId": 4,
                "documentsJson": "null",
                "documentIds": []
              },
              "addressType": {
                "createdBy": "null",
                "creationDate": "null",
                "lastModifiedBy": "null",
                "lastModifiedDate": "null",
                "id": 22,
                "type": "RA-BILLING"
              }
            },
            "firstName": "Mayuri",
            "middleName": "null",
            "lastName": "Chavan",
            "suffix": "null",
            "dateOfBirth": "null",
            "documentsJson": "null",
            "documentIds": [],
            "member_or_manager_value": "Member"
          },
          "Mom_Name": "Mayuri Chavan",
          "Address": {
            "MM_Address_Line_1": "507 B Amanora Mall",
            "MM_Address_Line_2": "",
            "MM_City": "Arizona",
            "MM_State": "AZ",
            "MM_Zip_Code": "11557"
          },
          "member_or_manger": "Member"
        },
        {
          "__temp_key__": 1,
          "Address": {
            "MM_State": "NJ"
          }
        }
      ]
    }
  }

form_fields = extract_form_fields(pdf_path)
pdf_text = extract_text_from_pdf(pdf_path)
field_context = extract_field_context(pdf_path, form_fields)
mapped_fields = get_mapped_fields(pdf_text, form_fields, field_context, payload)

print("Final AI Mappings:", mapped_fields)
fill_pdf_form(pdf_path, output_pdf_path, mapped_fields)
