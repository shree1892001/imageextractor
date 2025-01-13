from flask import Flask ,jsonify,request
from Extractor.Paddle import ImageTextExtractionService
from flask_cors import CORS
from Common.constants import *

db_connection_params = {
    "dbname": DB_NAME,
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASS
}
app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
service=ImageTextExtractionService(API_KEY,db_connection_params)
@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        data = request.get_json()
        image_path = data.get('image_path')

        if not image_path:
            return jsonify({"error": "Image path is required"}), 400

        table_name, flattened_data = service.process_image(image_path)
        return jsonify({"table_name": table_name, "flattened_data": flattened_data}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/data/<document_type>', methods=['GET'])
def get_data(document_type):
    try:
        data = service.get_data_by_document_type(document_type)
        if not data:
            return jsonify({"message": "No data found for the provided document type."}), 404
        return jsonify({"data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(debug=True,port=3002)