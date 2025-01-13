import os
import google.generativeai as genai
from IPython.display import Image as IPImage

class ImageTextExtractor:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._configure_api()

    def _configure_api(self):
        os.environ['GOOGLE_API_KEY'] = self.api_key
        genai.configure(api_key=self.api_key)

    def query_gemini_llm(self, image_path, prompt):
        ip_image = IPImage(filename=image_path)
        vision_model = genai.GenerativeModel(self.model_name)
        response = vision_model.generate_content([prompt, ip_image])
        return response.text