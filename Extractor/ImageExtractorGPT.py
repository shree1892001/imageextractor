import base64
import os
from openai import OpenAI
import json
from Common.constants import *

class ImageTextExtractor:
    """Class to handle image text extraction in different formats."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def image_to_base64(self, image_path: str):
        """Converts a local image to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_text_from_url(self, image_url: str):
        """Extracts text from an image using its URL."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text content from this image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    def extract_text_from_base64(self, image_base64: str):
        """Extracts text from an image using its base64 encoding."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all the text content from this image thoroughly and accurately. "
                                "Ensure that no lines, words, or parts of the content are missed, even if the text is faint, "
                                "small, or near the edges. The text may include headings, paragraphs, or lists and could appear "
                                "in various fonts, styles, or layouts. Carefully preserve the reading order and structure as it "
                                "appears in the image. Double-check for any skipped lines or incomplete content, and extract every "
                                "visible text element, ensuring completeness across all sections. This is crucial for the task's accuracy."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    def get_image_text(self, image_path: str):
        """Handles both image URL and local image file (jpg, jpeg, png formats)."""

        valid_extensions = ['jpg', 'jpeg', 'png']
        _, ext = os.path.splitext(image_path)
        ext = ext.lower().strip('.')

        if ext not in valid_extensions:
            return {"error": "Invalid image format. Please use JPG, JPEG, or PNG."}

        if image_path.startswith('http'):

            extracted_text = self.extract_text_from_url(image_path)
        else:

            image_base64 = self.image_to_base64(image_path)
            extracted_text = self.extract_text_from_base64(image_base64)

        return {"image_path": image_path, "extracted_text": extracted_text}

if __name__ == "__main__":

    api_key = OPENAI_KEY

    extractor = ImageTextExtractor(api_key)

    image_path = "D:\\TextExtractor\\Extractor\\images\\driver_license.jpg"

    result = extractor.get_image_text(image_path)

    print(json.dumps(result, indent=4))