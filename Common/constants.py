



DB_NAME = "test4"
DB_USER = "postgres"
DB_PASS  = "postgres"
DB_PORT= 5432
DB_SERVER_NAME="local_shreyas"
DB_HOST = "127.0.0.1"
DB_NAME = "test5"
CLASSIFICATION_PROMPT = """
Analyze this document image and identify the type of document by extracting only the following specific keywords if they are present: "license," "Pancard," or "aadharcard." , "passport," . Return the result in the following JSON format:        {
            "document_type": "The type of document (e.g., 'Pancard', 'License', 'AadhaarCard', etc.)",
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