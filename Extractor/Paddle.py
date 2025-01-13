import os
import json
from psycopg2 import sql
from IPython.display import Image as IPImage
import google.generativeai as genai
from utils.ApplicationConnection import ApplicationConnection


class ImageTextExtractor:
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._configure_api()

    def _configure_api(self):
        os.environ['GOOGLE_API_KEY'] = self.api_key
        genai.configure(api_key=self.api_key)

    def query_gemini_llm(self, image_path,
                         prompt="Extract the text data from the given image in a structured JSON format not as JSON string . Ensure the JSON output contains: document_type to classify the document (e.g., Aadhaar Card, PAN Card, Driving License, etc.). data with key-value pairs for all extracted fields (e.g., name, gender, date of birth, and other relevant details)."):
        ip_image = IPImage(filename=image_path)
        vision_model = genai.GenerativeModel(self.model_name)
        response = vision_model.generate_content([prompt, ip_image])
        return response.text


class DynamicDatabaseHandler:
    def __init__(self, app_connection):
        self.connection = app_connection.connect()
        self.connection.autocommit = False
        self.cursor = self.connection.cursor()

    def create_or_alter_table(self, table_name, json_data):
        table_name = table_name.translate(str.maketrans("", "", "'"))

        try:
            self.cursor.execute(sql.SQL("SELECT to_regclass(%s)"), [table_name])
            table_exists = self.cursor.fetchone()[0] is not None

            if not table_exists:
                columns = ", ".join([f"\"{key}\" TEXT" for key in json_data.keys()])
                create_table_query = sql.SQL(f"CREATE TABLE {table_name} ({columns})")
                self.cursor.execute(create_table_query)
                self.connection.commit()
            else:
                self.cursor.execute(
                    sql.SQL("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                    """), [table_name]
                )
                existing_columns = {row[0] for row in self.cursor.fetchall()}

                for key in json_data.keys():
                    if key not in existing_columns:
                        try:
                            alter_table_query = sql.SQL(
                                f'ALTER TABLE {table_name} ADD COLUMN "{key}" TEXT'
                            )
                            self.cursor.execute(alter_table_query)
                            self.connection.commit()
                        except Exception as e:
                            self.connection.rollback()
                            print(f"Warning: Could not add column {key}: {str(e)}")

        except Exception as e:
            self.connection.rollback()
            raise e

    def insert_data(self, table_name, json_data):
        try:
            for key, value in json_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    json_data[key] = json.dumps(value)

            keys = ", ".join([f'"{key}"' for key in json_data.keys()])
            placeholders = ", ".join([f"%({key})s" for key in json_data.keys()])
            insert_query = sql.SQL(f"INSERT INTO {table_name} ({keys}) VALUES ({placeholders})")
            self.cursor.execute(insert_query, json_data)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e

    def close(self):
        try:
            self.connection.commit()
        except Exception:
            self.connection.rollback()
        finally:
            self.cursor.close()
            self.connection.close()


def flatten_json(nested_json, parent_key='', sep='_'):
    items = []
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        new_key = new_key.replace(" ", "_").replace("'", "")
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, str(v)))
    return dict(items)


class ImageTextExtractionService:
    def __init__(self, api_key, db_connection_params):
        self.image_extractor = ImageTextExtractor(api_key)
        self.db_handler = DynamicDatabaseHandler(ApplicationConnection(**db_connection_params))

    def process_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        json_data = self.image_extractor.query_gemini_llm(image_path)
        clean_json_data = json_data.strip().replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(clean_json_data)

        document_type = parsed_data.get("document_type", "default")
        document_type = document_type.translate(str.maketrans("", "", "'"))
        table_name = f"extracted_data_{document_type.replace(' ', '_').lower()}"

        flattened_data = flatten_json(parsed_data)

        self.db_handler.create_or_alter_table(table_name, flattened_data)
        self.db_handler.insert_data(table_name, flattened_data)

        return table_name, flattened_data

    def get_data_by_document_type(self, document_type):
        table_name = f"extracted_data_{document_type.replace(' ', '_').lower()}"
        self.db_handler.cursor.execute(sql.SQL(f"SELECT * FROM {table_name}"))
        data = self.db_handler.cursor.fetchall()
        column_names = [desc[0] for desc in self.db_handler.cursor.description]
        result = [dict(zip(column_names, row)) for row in data]
        return result
