from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from Services.DocumentProcessor1 import DocumentProcessor  # Ensure this points to the correct path
from Common.constants import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

service = DocumentProcessor(API_KEY)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            logger.error("File path is required")
            return jsonify({"error": "File path is required"}), 400

        file_path = data.get('file_path')
        if not file_path:
            logger.error("File path is empty")
            return jsonify({"error": "File path is required"}), 400

        logger.info(f"Processing file: {file_path}")
        extracted_data = service.process_file(file_path)
        return jsonify(extracted_data), 200

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("An error occurred while processing the file")
        return jsonify({"error": str(e)}), 500

@app.route('/extract-images', methods=['POST'])
def extract_images():
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            logger.error("File path is required")
            return jsonify({"error": "File path is required"}), 400

        file_path = data.get('file_path')
        if not file_path:
            logger.error("File path is empty")
            return jsonify({"error": "File path is required"}), 400

        logger.info(f"Extracting images from file: {file_path}")
        extracted_images = service.extract_images(file_path)
        return jsonify(extracted_images), 200

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("An error occurred while extracting images from the file")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3002)