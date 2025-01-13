# import json
#
# # Input JSON string with triple quotes
# json_string = ```json
# {
#   "document_type": "Aadhaar Card",
#   "data": {
#     "Government_of_India": "भारत सरकार",
#     "Name": "नन्दलाल यादव / Nandlal Yadav",
#     "Gender": "पुरुष / MALE",
#     "Date_of_Birth": "01/01/1948",
#     "Aadhaar_Number": "4197 3443 7228",
#     "VID": "9141 8297 4137 4787",
#     "Download_Date": "11/04/2021",
#     "Issue_Date": "15/12/2014",
#     "Slogan": "मेरा आधार, मेरी पहचान"
#   }
# }
# ```
#
# # Load JSON string into a Python dictionary
# json_data = json.loads(json_string)
#
# # Convert back to a JSON string without triple quotes
# formatted_json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
#
# print(formatted_json_string)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_aadhaar_pdf(file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "AADHAAR CARD")
    c.drawString(100, 730, "Name: Sample Name")
    c.drawString(100, 710, "Aadhaar Number: 1234 5678 9012")
    c.drawString(100, 690, "Gender: Male")
    c.drawString(100, 670, "Date of Birth: 1990-01-01")
    c.drawString(100, 650, "Address: 123 Sample Street, Sample City, Sample State, 123456")
    c.save()

def create_pan_pdf(file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "PAN CARD")
    c.drawString(100, 730, "Name: Sample Name")
    c.drawString(100, 710, "Father's Name: Sample Father's Name")
    c.drawString(100, 690, "PAN Number: ABCDE1234F")
    c.drawString(100, 670, "Date of Birth: 1990-01-01")
    c.save()

def create_license_pdf(file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "DRIVING LICENSE")
    c.drawString(100, 730, "Name: Sample Name")
    c.drawString(100, 710, "License Number: DL-123456789012")
    c.drawString(100, 690, "Date of Birth: 1990-01-01")
    c.drawString(100, 670, "Address: 123 Sample Street, Sample City, Sample State, 123456")
    c.drawString(100, 650, "Valid From: 2010-01-01")
    c.drawString(100, 630, "Valid Until: 2030-01-01")
    c.drawString(100, 610, "Vehicle Categories: MCWG, LMV")
    c.save()

# Create sample PDFs
create_aadhaar_pdf("sample_aadhaar.pdf")
create_pan_pdf("sample_pan.pdf")
create_license_pdf("sample_license.pdf")

print("Sample PDFs created.")




