from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
import fitz
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from Common.constants import *

class ProcessingResult(BaseModel):
    status: str
    components: List[Dict[str, Any]]
    theme: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]


class PDFElement(BaseModel):
    id: str
    type: str
    content: Optional[str]
    position: Dict[str, float]
    properties: Dict[str, Any]


class PDFAnalyzer:
    """Generic PDF Analysis Class"""

    @staticmethod
    def extract_text_properties(text_span: Any) -> Dict[str, Any]:
        return {
            "font": text_span.get("font", "default"),
            "fontSize": text_span.get("size", 12),
            "color": text_span.get("color", "#000000"),
            "flags": text_span.get("flags", 0)
        }

    @staticmethod
    def extract_form_properties(field: Any) -> Dict[str, Any]:
        properties = {
            "required": getattr(field, "is_required", False),
            "readonly": getattr(field, "is_readonly", False),
            "defaultValue": getattr(field, "default_value", ""),
            "options": getattr(field, "choice_values", []),
            "fieldType": field.field_type_string
        }

        if field.field_type_string == "Text":
            properties.update({
                "maxLength": getattr(field, "max_len", 0),
                "multiline": getattr(field, "multiline", False)
            })
        elif field.field_type_string in ["Choice", "ListBox"]:
            properties.update({
                "multiSelect": getattr(field, "multiselect", False),
                "options": getattr(field, "choice_values", [])
            })

        return properties


class IntelligentPdfToReactAgent:
    def __init__(self, api_key: str):
        self.agent = Agent(
            model=GeminiModel("gemini-1.5-flash", api_key=api_key),
            system_prompt="""
            You are an expert AI system for converting any PDF to React components.
            Analyze the PDF structure and generate appropriate React components
            that maintain the exact layout, styling, and functionality of the original PDF.
            Handle all types of PDF elements including forms, text, images, and tables.
            """
        )
        self.analyzer = PDFAnalyzer()

    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response to get valid JSON"""
        try:
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            response_text = "\n".join([
                line for line in response_text.split("\n")
                if not line.strip().startswith("//")
            ])
            return response_text
        except Exception as e:
            print(f"Warning: Error cleaning AI response: {str(e)}")
            return response

    def _setup_output_directories(self, output_dir: str) -> Dict[str, str]:
        """Setup all necessary output directories"""
        directories = {
            'root': output_dir,
            'src': os.path.join(output_dir, 'src'),
            'components': os.path.join(output_dir, 'src', 'components'),
            'utils': os.path.join(output_dir, 'src', 'utils'),
            'assets': os.path.join(output_dir, 'src', 'assets'),
            'images': os.path.join(output_dir, 'src', 'assets', 'images'),
            'styles': os.path.join(output_dir, 'src', 'styles'),
            'types': os.path.join(output_dir, 'src', 'types'),
            'hooks': os.path.join(output_dir, 'src', 'hooks')
        }

        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)

        return directories

    async def analyze_pdf_structure(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Analyze any PDF structure and extract all elements"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            doc = fitz.open(pdf_path)
            directories = self._setup_output_directories(output_dir)

            pdf_structure = {
                "pages": [],
                "metadata": {
                    "totalPages": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "dimensions": {
                        "width": doc[0].rect.width,
                        "height": doc[0].rect.height
                    }
                }
            }

            for page_num in range(len(doc)):
                page = doc[page_num]
                elements = []

                # Extract form fields
                try:
                    for field in page.widgets():
                        element = PDFElement(
                            id=f"form_{len(elements)}",
                            type="form",
                            content=field.field_name,
                            position={
                                "x": field.rect.x0,
                                "y": field.rect.y0,
                                "width": field.rect.x1 - field.rect.x0,
                                "height": field.rect.y1 - field.rect.y0
                            },
                            properties=self.analyzer.extract_form_properties(field)
                        )
                        elements.append(element.model_dump())
                except Exception as form_error:
                    print(f"Warning: Error processing forms on page {page_num + 1}: {str(form_error)}")

                # Extract text blocks
                try:
                    for block in page.get_text("dict")["blocks"]:
                        if block.get("type") == 0:
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("text", "").strip():
                                        element = PDFElement(
                                            id=f"text_{len(elements)}",
                                            type="text",
                                            content=span.get("text", "").strip(),
                                            position={
                                                "x": span["bbox"][0],
                                                "y": span["bbox"][1],
                                                "width": span["bbox"][2] - span["bbox"][0],
                                                "height": span["bbox"][3] - span["bbox"][1]
                                            },
                                            properties=self.analyzer.extract_text_properties(span)
                                        )
                                        elements.append(element.model_dump())
                except Exception as text_error:
                    print(f"Warning: Error processing text on page {page_num + 1}: {str(text_error)}")

                # Extract images
                try:
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if base_image:
                                image_rect = page.get_image_bbox(xref)
                                if image_rect:
                                    # Save the image
                                    image_filename = f"image_{page_num}_{img_index}.{base_image['ext']}"
                                    image_path = os.path.join(directories['images'], image_filename)
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(base_image["image"])

                                    element = PDFElement(
                                        id=f"image_{len(elements)}",
                                        type="image",
                                        content=image_filename,  # Store the filename instead of empty string
                                        position={
                                            "x": image_rect.x0,
                                            "y": image_rect.y0,
                                            "width": image_rect.width,
                                            "height": image_rect.height
                                        },
                                        properties={
                                            "format": base_image["ext"],
                                            "width": base_image.get("width", 0),
                                            "height": base_image.get("height", 0),
                                            "colorspace": base_image.get("colorspace", "unknown"),
                                            "xref": xref,
                                            "image_index": img_index,
                                            "src": f"/assets/images/{image_filename}"
                                        }
                                    )
                                    elements.append(element.model_dump())
                        except Exception as img_error:
                            print(
                                f"Warning: Could not process image {img_index} on page {page_num + 1}: {str(img_error)}")
                            continue
                except Exception as images_error:
                    print(f"Warning: Error processing images on page {page_num + 1}: {str(images_error)}")

                # Extract tables
                try:
                    tables = page.find_tables()
                    if tables:
                        for table_num, table in enumerate(tables):
                            element = PDFElement(
                                id=f"table_{len(elements)}",
                                type="table",
                                content="",
                                position={
                                    "x": table.bbox[0],
                                    "y": table.bbox[1],
                                    "width": table.bbox[2] - table.bbox[0],
                                    "height": table.bbox[3] - table.bbox[1]
                                },
                                properties={
                                    "rows": len(table.cells),
                                    "cols": len(table.cells[0]) if table.cells else 0,
                                    "cells": [[cell.text for cell in row] for row in table.cells]
                                }
                            )
                            elements.append(element.model_dump())
                except Exception as table_error:
                    print(f"Warning: Error processing tables on page {page_num + 1}: {str(table_error)}")

                page_data = {
                    "pageNumber": page_num + 1,
                    "elements": elements
                }
                pdf_structure["pages"].append(page_data)

            doc.close()
            return pdf_structure

        except FileNotFoundError as e:
            print(f"Error: PDF file not found - {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error analyzing PDF: {str(e)}")
            return {"error": str(e)}

    async def generate_react_code(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate React components based on analysis"""
        try:
            generation_prompt = f"""
            Generate React components based on this PDF analysis:
            {json.dumps(analysis, indent=2)}

            Create components for all element types (text, forms, images, tables).
            Maintain exact positioning and styling.
            Use modern React best practices and TypeScript.
            Implement proper form handling and validation.
            Create responsive components where appropriate.

            Return ONLY a JSON object with this structure:
            {{
                "components": [
                    {{
                        "name": "string",
                        "code": "string (React component code)",
                        "styles": "string (styled-components code)",
                        "types": "string (TypeScript types)"
                    }}
                ],
                "theme": {{
                    "colors": object,
                    "typography": object,
                    "spacing": object
                }},
                "utils": {{
                    "validation": "string (validation utilities)",
                    "helpers": "string (helper functions)"
                }}
            }}
            """

            response = await self.agent.run(generation_prompt)
            cleaned_response = self._clean_ai_response(response.data)

            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"Error parsing generation response: {str(e)}")
                return {"components": [], "theme": {}, "utils": {}}

        except Exception as e:
            print(f"Error generating React code: {str(e)}")
            return {"error": str(e)}

    def save_react_project(self, output: Dict[str, Any], output_dir: str):
        """Save generated React project files"""
        try:
            directories = self._setup_output_directories(output_dir)

            # Save components
            for component in output.get("components", []):
                component_dir = os.path.join(directories['components'], component["name"])
                os.makedirs(component_dir, exist_ok=True)

                # Save component files
                for file_type, content in [
                    ("index.tsx", component["code"]),
                    ("styles.ts", component["styles"]),
                    ("types.ts", component["types"])
                ]:
                    with open(os.path.join(component_dir, file_type), "w", encoding='utf-8') as f:
                        f.write(content)

            # Save utilities
            utils = output.get("utils", {})
            for util_name, util_content in utils.items():
                with open(os.path.join(directories['utils'], f"{util_name}.ts"), "w", encoding='utf-8') as f:
                    f.write(util_content)

            # Save theme
            theme = output.get("theme", {})
            with open(os.path.join(directories['styles'], "theme.ts"), "w", encoding='utf-8') as f:
                f.write(f"export const theme = {json.dumps(theme, indent=2)}")

            # Save package.json
            package_json = {
                "name": "pdf-to-react-app",
                "version": "1.0.0",
                "private": True,
                "dependencies": {
                    "react": "^18.2.0",
                    "react-dom": "^18.2.0",
                    "styled-components": "^6.0.0",
                    "@types/react": "^18.2.0",
                    "@types/styled-components": "^6.0.0",
                    "react-scripts": "5.0.1",
                    "typescript": "^4.9.5",
                    "formik": "^2.4.0",
                    "yup": "^1.3.0",
                    "react-table": "^7.8.0",
                    "react-pdf": "^7.0.0",
                    "lodash": "^4.17.21"
                },
                "scripts": {
                    "start": "react-scripts start",
                    "build": "react-scripts build",
                    "test": "react-scripts test",
                    "eject": "react-scripts eject"
                }
            }

            with open(os.path.join(output_dir, "package.json"), "w", encoding='utf-8') as f:
                json.dump(package_json, f, indent=2)

            # Save tsconfig.json
            tsconfig = {
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "esModuleInterop": True,
                    "allowSyntheticDefaultImports": True,
                    "strict": True,
                    "forceConsistentCasingInFileNames": True,
                    "noFallthroughCasesInSwitch": True,
                    "module": "esnext",
                    "moduleResolution": "node",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "noEmit": True,
                    "jsx": "react-jsx"
                },
                "include": ["src"]
            }

            with open(os.path.join(output_dir, "tsconfig.json"), "w", encoding='utf-8') as f:
                json.dump(tsconfig, f, indent=2)

            print(f"\n📁 React project saved to: {output_dir}")

        except Exception as e:
            print(f"Error saving React project: {str(e)}")
            raise

    async def run(self, pdf_path: str, output_dir: str) -> ProcessingResult:
        """Main execution method"""
        start_time = datetime.now()

        try:
            print("\n🤖 Starting PDF to React Conversion...")

            # Validate PDF path
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Step 1: Analyze PDF
            print("\n📄 Analyzing PDF structure...")
            analysis = await self.analyze_pdf_structure(pdf_path, output_dir)
            if "error" in analysis:
                raise Exception(analysis["error"])

            # Step 2: Generate React code
            print("\n💻 Generating React components...")
            output = await self.generate_react_code(analysis)
            if "error" in output:
                raise Exception(output["error"])

            # Step 3: Save React project
            print("\n💾 Saving React project...")
            self.save_react_project(output, output_dir)

            result = ProcessingResult(
                status="success",
                components=output.get("components", []),
                theme=output.get("theme", {}),
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "pdf_name": os.path.basename(pdf_path),
                    "total_pages": analysis["metadata"]["totalPages"],
                    "generated_at": datetime.now().isoformat(),
                    "components_count": len(output.get("components", [])),
                }
            )

            print(f"\n✅ React project generated successfully!")
            print(f"📁 Output directory: {output_dir}")
            print(f"⏱️ Execution time: {result.execution_time:.2f} seconds")
            print(f"📦 Components generated: {len(result.components)}")
            print(f"📄 Total pages processed: {analysis['metadata']['totalPages']}")

            return result

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return ProcessingResult(
                status="error",
                components=[],
                theme={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "error": str(e),
                    "pdf_name": os.path.basename(pdf_path),
                    "generated_at": datetime.now().isoformat()
                }
            )


async def main():
    api_key = API_KEY # Replace with your API key
    agent = IntelligentPdfToReactAgent(api_key)

    # Use absolute path to PDF file
    pdf_path = os.path.abspath("D:\\demo\\Services\\document-677cb8e821dd98.25791359.pdf")
    output_dir = "react_output2"

    result = await agent.run(
        pdf_path=pdf_path,
        output_dir=output_dir
    )

    if result.status == "success":
        print("✅ PDF successfully converted to React application!")
        print(f"Generated {len(result.components)} components")
        print(f"Processing time: {result.execution_time:.2f} seconds")
    else:
        print("❌ Error during conversion")
        print(f"Error: {result.metadata.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())