# # import asyncio
# # import json
# # import os
# # import shutil
# # import uuid
# # from typing import Dict, Any, List, Optional
# # import fitz
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
# # from fastapi.responses import JSONResponse, FileResponse
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field
# #
# # from Services.GenericFiller import MultiAgentFormFiller as StandardFormFiller
# # from Services.GenFiler import MultiAgentFormFiller as OCRFormFiller
# #
# # app = FastAPI(title="Smart PDF Form Filler API",
# #               description="API for intelligent PDF form filling with automatic OCR detection")
# #
# # origins = ['*']
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# #
# # class FieldMatch(BaseModel):
# #     json_field: str = Field(..., description="Field from input JSON")
# #     pdf_field: str = Field(..., description="Corresponding PDF form field")
# #     confidence: float = Field(..., ge=0, le=1, description="Matching confidence")
# #     suggested_value: Any = Field(None, description="Value suggested for the field")
# #     reasoning: str = Field(None, description="Reasoning behind the match")
# #
# # class FormResponse(BaseModel):
# #     status: str = Field(..., description="Processing status")
# #     message: str = Field(..., description="Detailed processing message")
# #     requires_ocr: bool = Field(False, description="Whether OCR was required")
# #     file_path: Optional[str] = Field(None, description="Path to processed file")
# #     field_matches: List[FieldMatch] = Field(default_factory=list, description="Field matching details")
# #     error_details: Optional[str] = Field(None, description="Error details if processing failed")
# #
# # class PDFProcessingService:
# #     @staticmethod
# #     async def analyze_form_fields(pdf_path: str) -> tuple[bool, list]:
# #         """Advanced form field analysis with improved OCR detection logic"""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             fields = []
# #             total_interactive_fields = 0
# #             total_pages = len(doc)
# #
# #             filled_fields = 0
# #             empty_fields = 0
# #
# #             for page in doc:
# #                 for widget in page.widgets():
# #                     if widget.field_name:
# #                         total_interactive_fields += 1
# #                         field_value = widget.field_value
# #
# #                         if field_value:
# #                             filled_fields += 1
# #                         else:
# #                             empty_fields += 1
# #
# #                         fields.append({
# #                             'name': widget.field_name,
# #                             'type': widget.field_type,
# #                             'value': field_value
# #                         })
# #
# #             doc.close()
# #
# #             needs_ocr = (
# #                 total_interactive_fields == 0 or
# #                 (total_pages > 2 and total_interactive_fields < 3) or
# #                 (total_interactive_fields > 0 and empty_fields == total_interactive_fields)
# #             )
# #
# #             print(f"OCR Detection Analysis: pages={total_pages}, fields={total_interactive_fields}, "
# #                   f"filled={filled_fields}, empty={empty_fields}, needs_ocr={needs_ocr}")
# #
# #             return needs_ocr, fields
# #
# #         except Exception as e:
# #             print(f"Field analysis error: {e}")
# #             return True, []
# #
# # class TemporaryFileManager:
# #     @staticmethod
# #     def generate_unique_filename(prefix: str = "", extension: str = "") -> str:
# #         """Generate a unique filename with optional prefix and extension"""
# #         unique_id = str(uuid.uuid4())
# #         return f"{prefix}{unique_id}{extension}"
# #
# #     @staticmethod
# #     def cleanup_file(file_path: str):
# #         """Safe file cleanup with error handling"""
# #         try:
# #             if os.path.exists(file_path):
# #                 os.remove(file_path)
# #                 print(f"Cleaned up file: {file_path}")
# #         except Exception as e:
# #             print(f"File cleanup error for {file_path}: {e}")
# #
# # class PDFFormFillerAPI:
# #     def __init__(self):
# #         self.app = FastAPI(
# #             title="Intelligent PDF Form Filler API",
# #             description="Advanced API for intelligent PDF form processing"
# #         )
# #
# #         self.app.add_middleware(
# #             CORSMiddleware,
# #             allow_origins=["*"],
# #             allow_credentials=True,
# #             allow_methods=["*"],
# #             allow_headers=["*"]
# #         )
# #
# #         self.setup_routes()
# #
# #     def setup_routes(self):
# #         @self.app.post("/api/process-form", response_model=FormResponse)
# #         async def process_form(
# #                 pdf_file: UploadFile = File(...),
# #                 form_data: str = Form(...),
# #                 force_ocr: bool = Form(False),
# #                 return_json: bool = Form(False),
# #                 background_tasks: BackgroundTasks = BackgroundTasks()
# #         ):
# #             temp_input_path = ""
# #             output_path = ""
# #             permanent_path = ""
# #
# #             try:
# #
# #                 print(f"Processing form request: filename={pdf_file.filename}, force_ocr={force_ocr}")
# #
# #                 temp_input_path = TemporaryFileManager.generate_unique_filename(
# #                     prefix="temp_input_", extension=".pdf"
# #                 )
# #                 output_path = TemporaryFileManager.generate_unique_filename(
# #                     prefix="filled_", extension=".pdf"
# #                 )
# #
# #                 with open(temp_input_path, "wb") as f:
# #                     content = await pdf_file.read()
# #                     f.write(content)
# #
# #                 json_data = json.loads(form_data)
# #
# #                 needs_ocr, fields = await PDFProcessingService.analyze_form_fields(temp_input_path)
# #
# #                 final_ocr_decision = needs_ocr or force_ocr
# #
# #                 print(f"Final OCR decision: auto_detect={needs_ocr}, force_ocr={force_ocr}, "
# #                       f"using_ocr={final_ocr_decision}, fields_count={len(fields)}")
# #
# #                 form_filler = OCRFormFiller() if final_ocr_decision else StandardFormFiller()
# #
# #                 print(f"Selected form filler: {'OCR' if final_ocr_decision else 'Standard'}")
# #
# #                 result = await form_filler.match_and_fill_fields(
# #                     temp_input_path, json_data, output_path
# #                 )
# #
# #                 if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
# #                     raise ValueError("Failed to generate filled PDF")
# #
# #                 os.makedirs("filled_forms", exist_ok=True)
# #                 permanent_path = os.path.join(
# #                     "filled_forms",
# #                     TemporaryFileManager.generate_unique_filename(
# #                         prefix="filled_", extension=".pdf"
# #                     )
# #                 )
# #                 shutil.copy2(output_path, permanent_path)
# #
# #                 print(f"Form successfully filled: output={permanent_path}")
# #
# #                 field_matches = [
# #                     FieldMatch(
# #                         json_field="processed_form",
# #                         pdf_field="system_confirmation",
# #                         confidence=1.0,
# #                         suggested_value=True,
# #                         reasoning="Successful form processing"
# #                     )
# #                 ]
# #
# #                 background_tasks.add_task(TemporaryFileManager.cleanup_file, temp_input_path)
# #                 background_tasks.add_task(TemporaryFileManager.cleanup_file, output_path)
# #
# #                 if return_json:
# #                     return FormResponse(
# #                         status="success",
# #                         message="Form processed successfully",
# #                         requires_ocr=final_ocr_decision,
# #                         file_path=permanent_path,
# #                         field_matches=field_matches
# #                     )
# #                 else:
# #                     return FileResponse(
# #                         path=permanent_path,
# #                         filename=f"filled_{pdf_file.filename}",
# #                         media_type="application/pdf"
# #                     )
# #
# #             except Exception as e:
# #
# #                 error_msg = str(e)
# #                 print(f"Form processing error: {error_msg}")
# #
# #                 error_response = FormResponse(
# #                     status="error",
# #                     message=f"Processing failed: {error_msg}",
# #                     error_details=error_msg,
# #                     requires_ocr=False
# #                 )
# #
# #                 for path in [temp_input_path, output_path]:
# #                     if path and os.path.exists(path):
# #                         background_tasks.add_task(TemporaryFileManager.cleanup_file, path)
# #
# #                 raise HTTPException(status_code=500, detail=error_response.dict())
# #
# #     def get_app(self):
# #         return self.app
# #
# # pdf_form_filler_api = PDFFormFillerAPI()
# # app = pdf_form_filler_api.get_app()
# #
# # if __name__ == "__main__":
# #     import uvicorn
# #
# #     uvicorn.run(app, host="0.0.0.0", port=8005)
# import asyncio
#
# #
# # import json
# # import os
# # import shutil
# # import uuid
# # from typing import Dict, Any, List, Optional
# # import fitz
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
# # from fastapi.responses import JSONResponse, FileResponse
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field
# #
# # from Services.GenericFiller import MultiAgentFormFiller as StandardFormFiller
# # from Services.GenFiler import MultiAgentFormFiller as OCRFormFiller
# #
# # app = FastAPI(title="Smart PDF Form Filler API",
# #               description="API for intelligent PDF form filling with automatic OCR detection")
# #
# # # ✅ List of allowed frontend URLs
# # ALLOWED_ORIGINS = [
# #     "https://www.redberyltest.in",  # Production
# #     "https://staging.redberyltest.in",  # Staging
# #     "http://localhost:3000",  # Local Dev
# #     "http://127.0.0.1:3000"
# # ]
# #
# # # ✅ Apply CORS Middleware with specific origins
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=ALLOWED_ORIGINS,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# #
# # class FieldMatch(BaseModel):
# #     json_field: str = Field(..., description="Field from input JSON")
# #     pdf_field: str = Field(..., description="Corresponding PDF form field")
# #     confidence: float = Field(..., ge=0, le=1, description="Matching confidence")
# #     suggested_value: Any = Field(None, description="Value suggested for the field")
# #     reasoning: str = Field(None, description="Reasoning behind the match")
# #
# # class FormResponse(BaseModel):
# #     status: str = Field(..., description="Processing status")
# #     message: str = Field(..., description="Detailed processing message")
# #     requires_ocr: bool = Field(False, description="Whether OCR was required")
# #     file_path: Optional[str] = Field(None, description="Path to processed file")
# #     field_matches: List[FieldMatch] = Field(default_factory=list, description="Field matching details")
# #     error_details: Optional[str] = Field(None, description="Error details if processing failed")
# #
# # class PDFProcessingService:
# #     @staticmethod
# #     async def analyze_form_fields(pdf_path: str) -> tuple[bool, list]:
# #         """Advanced form field analysis with improved OCR detection logic"""
# #         try:
# #             doc = fitz.open(pdf_path)
# #             fields = []
# #             total_interactive_fields = 0
# #             total_pages = len(doc)
# #
# #             filled_fields = 0
# #             empty_fields = 0
# #
# #             for page in doc:
# #                 for widget in page.widgets():
# #                     if widget.field_name:
# #                         total_interactive_fields += 1
# #                         field_value = widget.field_value
# #
# #                         if field_value:
# #                             filled_fields += 1
# #                         else:
# #                             empty_fields += 1
# #
# #                         fields.append({
# #                             'name': widget.field_name,
# #                             'type': widget.field_type,
# #                             'value': field_value
# #                         })
# #
# #             doc.close()
# #
# #             needs_ocr = (
# #                 total_interactive_fields == 0 or
# #                 (total_pages > 2 and total_interactive_fields < 3) or
# #                 (total_interactive_fields > 0 and empty_fields == total_interactive_fields)
# #             )
# #
# #             print(f"OCR Detection Analysis: pages={total_pages}, fields={total_interactive_fields}, "
# #                   f"filled={filled_fields}, empty={empty_fields}, needs_ocr={needs_ocr}")
# #
# #             return needs_ocr, fields
# #
# #         except Exception as e:
# #             print(f"Field analysis error: {e}")
# #             return True, []
# #
# # class TemporaryFileManager:
# #     @staticmethod
# #     def generate_unique_filename(prefix: str = "", extension: str = "") -> str:
# #         """Generate a unique filename with optional prefix and extension"""
# #         unique_id = str(uuid.uuid4())
# #         return f"{prefix}{unique_id}{extension}"
# #
# #     @staticmethod
# #     def cleanup_file(file_path: str):
# #         """Safe file cleanup with error handling"""
# #         try:
# #             if os.path.exists(file_path):
# #                 os.remove(file_path)
# #                 print(f"Cleaned up file: {file_path}")
# #         except Exception as e:
# #             print(f"File cleanup error for {file_path}: {e}")
# #
# # class PDFFormFillerAPI:
# #     def __init__(self):
# #         self.app = FastAPI(
# #             title="Intelligent PDF Form Filler API",
# #             description="Advanced API for intelligent PDF form processing"
# #         )
# #
# #         # ✅ Apply CORS Middleware with multiple origins
# #         self.app.add_middleware(
# #             CORSMiddleware,
# #             allow_origins=ALLOWED_ORIGINS,
# #             allow_credentials=True,
# #             allow_methods=["*"],
# #             allow_headers=["*"]
# #         )
# #
# #         self.setup_routes()
# #
# #     def setup_routes(self):
# #         @self.app.post("/api/process-form", response_model=FormResponse)
# #         async def process_form(
# #                 pdf_file: UploadFile = File(...),
# #                 form_data: str = Form(...),
# #                 force_ocr: bool = Form(False),
# #                 return_json: bool = Form(False),
# #                 background_tasks: BackgroundTasks = BackgroundTasks()
# #         ):
# #             temp_input_path = ""
# #             output_path = ""
# #             permanent_path = ""
# #
# #             try:
# #                 print(f"Processing form request: filename={pdf_file.filename}, force_ocr={force_ocr}")
# #
# #                 temp_input_path = TemporaryFileManager.generate_unique_filename(
# #                     prefix="temp_input_", extension=".pdf"
# #                 )
# #                 output_path = TemporaryFileManager.generate_unique_filename(
# #                     prefix="filled_", extension=".pdf"
# #                 )
# #
# #                 with open(temp_input_path, "wb") as f:
# #                     content = await pdf_file.read()
# #                     f.write(content)
# #
# #                 json_data = json.loads(form_data)
# #
# #                 needs_ocr, fields = await PDFProcessingService.analyze_form_fields(temp_input_path)
# #                 final_ocr_decision = needs_ocr or force_ocr
# #
# #                 form_filler = OCRFormFiller() if final_ocr_decision else StandardFormFiller()
# #                 result = await form_filler.match_and_fill_fields(temp_input_path, json_data, output_path)
# #
# #                 if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
# #                     raise ValueError("Failed to generate filled PDF")
# #
# #                 os.makedirs("filled_forms", exist_ok=True)
# #                 permanent_path = os.path.join(
# #                     "filled_forms",
# #                     TemporaryFileManager.generate_unique_filename(prefix="filled_", extension=".pdf")
# #                 )
# #                 shutil.copy2(output_path, permanent_path)
# #
# #                 print(f"Form successfully filled: output={permanent_path}")
# #
# #                 field_matches = [
# #                     FieldMatch(
# #                         json_field="processed_form",
# #                         pdf_field="system_confirmation",
# #                         confidence=1.0,
# #                         suggested_value=True,
# #                         reasoning="Successful form processing"
# #                     )
# #                 ]
# #
# #                 background_tasks.add_task(TemporaryFileManager.cleanup_file, temp_input_path)
# #                 background_tasks.add_task(TemporaryFileManager.cleanup_file, output_path)
# #
# #                 if return_json:
# #                     return FormResponse(
# #                         status="success",
# #                         message="Form processed successfully",
# #                         requires_ocr=final_ocr_decision,
# #                         file_path=permanent_path,
# #                         field_matches=field_matches
# #                     )
# #                 else:
# #                     return FileResponse(
# #                         path=permanent_path,
# #                         filename=f"filled_{pdf_file.filename}",
# #                         media_type="application/pdf"
# #                     )
# #
# #             except Exception as e:
# #                 error_msg = str(e)
# #                 print(f"Form processing error: {error_msg}")
# #                 raise HTTPException(status_code=500, detail={"error": error_msg})
# #
# #     def get_app(self):
# #         return self.app
# #
# # pdf_form_filler_api = PDFFormFillerAPI()
# # app = pdf_form_filler_api.get_app()
# #
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8005)
#
#
# import asyncio
# import json
# import os
# import shutil
# import uuid
# from typing import Dict, Any, List, Optional
# import fitz
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware  # Import HTTPS redirect middleware
# from pydantic import BaseModel, Field
#
# from Services.GenericFiller import MultiAgentFormFiller as StandardFormFiller
# from Services.GenFiler import MultiAgentFormFiller as OCRFormFiller
#
# app = FastAPI(
#     title="Smart PDF Form Filler API",
#     description="API for intelligent PDF form filling with automatic OCR detection"
# )
#
# ALLOWED_ORIGINS = [
#     "https://www.redberyltest.in",
#     "https://staging.redberyltest.in",
#     "http://localhost:3000",
#     "http://127.0.0.1:3000"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Add middleware to force HTTPS redirection
# app.add_middleware(HTTPSRedirectMiddleware)
#
#
# # ... [Rest of your code remains unchanged] ...
#
# class FieldMatch(BaseModel):
#     json_field: str = Field(..., description="Field from input JSON")
#     pdf_field: str = Field(..., description="Corresponding PDF form field")
#     confidence: float = Field(..., ge=0, le=1, description="Matching confidence")
#     suggested_value: Any = Field(None, description="Value suggested for the field")
#     reasoning: str = Field(None, description="Reasoning behind the match")
#
#
# class FormResponse(BaseModel):
#     status: str = Field(..., description="Processing status")
#     message: str = Field(..., description="Detailed processing message")
#     requires_ocr: bool = Field(False, description="Whether OCR was required")
#     file_path: Optional[str] = Field(None, description="Path to processed file")
#     field_matches: List[FieldMatch] = Field(default_factory=list, description="Field matching details")
#     error_details: Optional[str] = Field(None, description="Error details if processing failed")
#
#
# class PDFProcessingService:
#     @staticmethod
#     async def analyze_form_fields(pdf_path: str) -> tuple[bool, list]:
#         try:
#             doc = fitz.open(pdf_path)
#             fields = []
#             total_interactive_fields = 0
#             total_pages = len(doc)
#             filled_fields = 0
#             empty_fields = 0
#
#             for page in doc:
#                 for widget in page.widgets():
#                     if widget.field_name:
#                         total_interactive_fields += 1
#                         field_value = widget.field_value
#                         if field_value:
#                             filled_fields += 1
#                         else:
#                             empty_fields += 1
#                         fields.append({
#                             'name': widget.field_name,
#                             'type': widget.field_type,
#                             'value': field_value
#                         })
#
#             doc.close()
#             needs_ocr = (
#                     total_interactive_fields == 0 or
#                     (total_pages > 2 and total_interactive_fields < 3) or
#                     (total_interactive_fields > 0 and empty_fields == total_interactive_fields)
#             )
#
#             print(f"OCR Detection Analysis: pages={total_pages}, fields={total_interactive_fields}, "
#                   f"filled={filled_fields}, empty={empty_fields}, needs_ocr={needs_ocr}")
#
#             return needs_ocr, fields
#
#         except Exception as e:
#             print(f"Field analysis error: {e}")
#             return True, []
#
#
# class TemporaryFileManager:
#     @staticmethod
#     def generate_unique_filename(prefix: str = "", extension: str = "") -> str:
#         unique_id = str(uuid.uuid4())
#         return f"{prefix}{unique_id}{extension}"
#
#     @staticmethod
#     def cleanup_file(file_path: str):
#         try:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 print(f"Cleaned up file: {file_path}")
#         except Exception as e:
#             print(f"File cleanup error for {file_path}: {e}")
#
#
# class PDFFormFillerAPI:
#     def __init__(self):
#         self.app = FastAPI(
#             title="Intelligent PDF Form Filler API",
#             description="Advanced API for intelligent PDF form processing"
#         )
#
#         self.app.add_middleware(
#             CORSMiddleware,
#             allow_origins=ALLOWED_ORIGINS,
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"]
#         )
#         # Enforce HTTPS redirection on this app instance as well
#         self.app.add_middleware(HTTPSRedirectMiddleware)
#
#         self.setup_routes()
#
#     def setup_routes(self):
#         @self.app.post("/api/process-form", response_model=FormResponse)
#         async def process_form(
#                 pdf_file: UploadFile = File(...),
#                 form_data: str = Form(...),
#                 force_ocr: bool = Form(False),
#                 return_json: bool = Form(False),
#                 background_tasks: BackgroundTasks = BackgroundTasks()
#         ):
#             temp_input_path = ""
#             output_path = ""
#             permanent_path = ""
#
#             try:
#                 print(f"Processing form request: filename={pdf_file.filename}, force_ocr={force_ocr}")
#
#                 temp_input_path = TemporaryFileManager.generate_unique_filename(prefix="temp_input_", extension=".pdf")
#                 output_path = TemporaryFileManager.generate_unique_filename(prefix="filled_", extension=".pdf")
#
#                 with open(temp_input_path, "wb") as f:
#                     content = await pdf_file.read()
#                     f.write(content)
#
#                 json_data = json.loads(form_data)
#
#                 needs_ocr, fields = await PDFProcessingService.analyze_form_fields(temp_input_path)
#                 final_ocr_decision = needs_ocr or force_ocr
#
#                 form_filler = OCRFormFiller() if final_ocr_decision else StandardFormFiller()
#                 result = await form_filler.match_and_fill_fields(temp_input_path, json_data, output_path)
#
#                 if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
#                     raise ValueError("Failed to generate filled PDF")
#
#                 os.makedirs("filled_forms", exist_ok=True)
#                 permanent_path = os.path.join(
#                     "filled_forms",
#                     TemporaryFileManager.generate_unique_filename(prefix="filled_", extension=".pdf")
#                 )
#                 shutil.copy2(output_path, permanent_path)
#
#                 print(f"Form successfully filled: output={permanent_path}")
#
#                 field_matches = [
#                     FieldMatch(
#                         json_field="processed_form",
#                         pdf_field="system_confirmation",
#                         confidence=1.0,
#                         suggested_value=True,
#                         reasoning="Successful form processing"
#                     )
#                 ]
#
#                 background_tasks.add_task(TemporaryFileManager.cleanup_file, temp_input_path)
#                 background_tasks.add_task(TemporaryFileManager.cleanup_file, output_path)
#
#                 if return_json:
#                     return FormResponse(
#                         status="success",
#                         message="Form processed successfully",
#                         requires_ocr=final_ocr_decision,
#                         file_path=permanent_path,
#                         field_matches=field_matches
#                     )
#                 else:
#                     return FileResponse(
#                         path=permanent_path,
#                         filename=f"filled_{pdf_file.filename}",
#                         media_type="application/pdf"
#                     )
#
#             except Exception as e:
#                 error_msg = str(e)
#                 print(f"Form processing error: {error_msg}")
#                 raise HTTPException(status_code=500, detail={"error": error_msg})
#
#     def get_app(self):
#         return self.app
#
#
# pdf_form_filler_api = PDFFormFillerAPI()
# app = pdf_form_filler_api.get_app()
#
# if __name__ == "__main__":
#     import uvicorn
#
#     # Specify the paths to your SSL certificate and key files
#     ssl_certfile = "D:\\demo\\cert.pem"
#     ssl_keyfile = "D:\\demo\\key.pem"
#
#     uvicorn.run(app, host="0.0.0.0", port=8005, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
import asyncio
import json
import os
import shutil
import uuid
from typing import Dict, Any, List, Optional
import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel, Field

from Services.GenericFiller import MultiAgentFormFiller as StandardFormFiller
from Services.GenFiler import MultiAgentFormFiller as OCRFormFiller

app = FastAPI(
    title="Smart PDF Form Filler API",
    description="API for intelligent PDF form filling with automatic OCR detection"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to force HTTPS redirection
app.add_middleware(HTTPSRedirectMiddleware)


class FieldMatch(BaseModel):
    json_field: str = Field(..., description="Field from input JSON")
    pdf_field: str = Field(..., description="Corresponding PDF form field")
    confidence: float = Field(..., ge=0, le=1, description="Matching confidence")
    suggested_value: Any = Field(None, description="Value suggested for the field")
    reasoning: str = Field(None, description="Reasoning behind the match")


class FormResponse(BaseModel):
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Detailed processing message")
    requires_ocr: bool = Field(False, description="Whether OCR was required")
    file_path: Optional[str] = Field(None, description="Path to processed file")
    field_matches: List[FieldMatch] = Field(default_factory=list, description="Field matching details")
    error_details: Optional[str] = Field(None, description="Error details if processing failed")


class PDFProcessingService:
    @staticmethod
    async def analyze_form_fields(pdf_path: str) -> tuple[bool, list]:
        try:
            doc = fitz.open(pdf_path)
            fields = []
            total_interactive_fields = 0
            total_pages = len(doc)
            filled_fields = 0
            empty_fields = 0

            for page in doc:
                for widget in page.widgets():
                    if widget.field_name:
                        total_interactive_fields += 1
                        field_value = widget.field_value
                        if field_value:
                            filled_fields += 1
                        else:
                            empty_fields += 1
                        fields.append({
                            'name': widget.field_name,
                            'type': widget.field_type,
                            'value': field_value
                        })

            doc.close()
            needs_ocr = (
                    total_interactive_fields == 0 or
                    (total_pages > 2 and total_interactive_fields < 3) or
                    (total_interactive_fields > 0 and empty_fields == total_interactive_fields)
            )

            print(f"OCR Detection Analysis: pages={total_pages}, fields={total_interactive_fields}, "
                  f"filled={filled_fields}, empty={empty_fields}, needs_ocr={needs_ocr}")

            return needs_ocr, fields

        except Exception as e:
            print(f"Field analysis error: {e}")
            return True, []


class TemporaryFileManager:
    @staticmethod
    def generate_unique_filename(prefix: str = "", extension: str = "") -> str:
        unique_id = str(uuid.uuid4())
        return f"{prefix}{unique_id}{extension}"

    @staticmethod
    def cleanup_file(file_path: str):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
        except Exception as e:
            print(f"File cleanup error for {file_path}: {e}")


class StateFormManager:
    """Manages state-specific form templates"""

    # Base directory for state form templates
    TEMPLATE_DIR = "D:\\demo\\state_templates"

    @staticmethod
    def get_state_form_path(entity_type: str, state: str) -> str:

        # Normalize inputs
        entity_type = entity_type.lower().strip()
        state = state.upper().strip()

        # Check if state directory exists
        state_dir = os.path.join(StateFormManager.TEMPLATE_DIR, state)
        if not os.path.exists(state_dir):
            raise FileNotFoundError(f"No forms available for state: {state}")

        # List all PDFs in the state directory to find matching forms
        available_forms = [f for f in os.listdir(state_dir) if f.lower().endswith('.pdf')]

        # Try to find a form that matches the entity type
        matching_forms = [f for f in available_forms if entity_type.lower() in f.lower()]

        if not matching_forms:
            raise FileNotFoundError(f"No form template found for {entity_type} in {state}")

        # Use the first matching form
        return os.path.join(state_dir, matching_forms[0])


class PDFFormFillerAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Intelligent PDF Form Filler API",
            description="Advanced API for intelligent PDF form processing"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        # Enforce HTTPS redirection on this app instance as well
        self.app.add_middleware(HTTPSRedirectMiddleware)

        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/api/process-form", response_model=FormResponse)
        async def process_form(
                pdf_file: Optional[UploadFile] = File(None),
                form_data: str = Form(...),
                force_ocr: bool = Form(False),
                return_json: bool = Form(False),
                background_tasks: BackgroundTasks = BackgroundTasks()
        ):
            temp_input_path = ""
            output_path = ""
            permanent_path = ""
            selected_template = None

            try:
                json_data = json.loads(form_data)

                # Get entity type and state from JSON
                entity_type = json_data.get("entity_type")
                state = json_data.get("state")

                # Check if both entity_type and state are provided
                if entity_type and state:
                    try:
                        # Try to get state form based on entity type and state
                        state_form_path = StateFormManager.get_state_form_path(entity_type, state)
                        selected_template = os.path.basename(state_form_path)

                        print(f"Using state form template: {state_form_path} for {entity_type} in {state}")

                        # Use state form as input file
                        temp_input_path = TemporaryFileManager.generate_unique_filename(
                            prefix="temp_input_", extension=".pdf"
                        )
                        shutil.copy2(state_form_path, temp_input_path)
                    except FileNotFoundError as e:
                        print(f"State form error: {e}")
                        # If uploaded file is available, use it as fallback
                        if pdf_file:
                            print(f"Falling back to uploaded file: {pdf_file.filename}")
                        else:
                            raise ValueError(
                                f"No state form template found for {entity_type} in {state} and no file was uploaded")

                # If no entity_type/state or form wasn't found, use uploaded file
                if not temp_input_path and pdf_file:
                    print(f"Processing uploaded file: filename={pdf_file.filename}, force_ocr={force_ocr}")

                    temp_input_path = TemporaryFileManager.generate_unique_filename(
                        prefix="temp_input_", extension=".pdf"
                    )

                    with open(temp_input_path, "wb") as f:
                        content = await pdf_file.read()
                        f.write(content)

                # Create output path
                output_path = TemporaryFileManager.generate_unique_filename(
                    prefix="filled_", extension=".pdf"
                )

                # Process the file
                needs_ocr, fields = await PDFProcessingService.analyze_form_fields(temp_input_path)
                final_ocr_decision = needs_ocr or force_ocr

                form_filler = OCRFormFiller() if final_ocr_decision else StandardFormFiller()
                result = await form_filler.match_and_fill_fields(temp_input_path, json_data, output_path)

                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise ValueError("Failed to generate filled PDF")

                os.makedirs("filled_forms", exist_ok=True)
                permanent_path = os.path.join(
                    "filled_forms",
                    TemporaryFileManager.generate_unique_filename(prefix="filled_", extension=".pdf")
                )
                shutil.copy2(output_path, permanent_path)

                # Get output filename from JSON if available, otherwise use default naming
                output_filename = json_data.get("filename", None)

                if not output_filename:
                    # Use selected template name if available
                    if selected_template:
                        output_filename = f"filled_{selected_template}"
                    # Fallback to uploaded filename if available
                    elif pdf_file:
                        output_filename = f"filled_{pdf_file.filename}"
                    # Last resort generic name
                    else:
                        output_filename = f"filled_form_{uuid.uuid4()}.pdf"

                # Ensure filename has .pdf extension
                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'

                print(f"Form successfully filled: output={permanent_path}, filename={output_filename}")

                field_matches = [
                    FieldMatch(
                        json_field="processed_form",
                        pdf_field="system_confirmation",
                        confidence=1.0,
                        suggested_value=True,
                        reasoning="Successful form processing"
                    )
                ]

                background_tasks.add_task(TemporaryFileManager.cleanup_file, temp_input_path)
                background_tasks.add_task(TemporaryFileManager.cleanup_file, output_path)

                if return_json:
                    return FormResponse(
                        status="success",
                        message="Form processed successfully",
                        requires_ocr=final_ocr_decision,
                        file_path=permanent_path,
                        field_matches=field_matches
                    )
                else:
                    return FileResponse(
                        path=permanent_path,
                        filename=output_filename,
                        media_type="application/pdf"
                    )

            except Exception as e:
                error_msg = str(e)
                print(f"Form processing error: {error_msg}")

                # Clean up any temporary files
                for path in [temp_input_path, output_path]:
                    if path and os.path.exists(path):
                        background_tasks.add_task(TemporaryFileManager.cleanup_file, path)

                raise HTTPException(status_code=500, detail={"error": error_msg})

    def get_app(self):
        return self.app


pdf_form_filler_api = PDFFormFillerAPI()
app = pdf_form_filler_api.get_app()

if __name__ == "__main__":
    import uvicorn

    # Specify the paths to your SSL certificate and key files


    uvicorn.run(app, host="0.0.0.0", port=8005, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)