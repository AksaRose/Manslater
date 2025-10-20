from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

api_url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text_to_translate = data.get('text', '')

    if not text_to_translate:
        return jsonify({"error": "No text provided for translation"}), 400

    payload = {
        "model": FINE_TUNED_MODEL,
        "messages": [
            {"role": "system", "content": "You are Manslater, an AI that translates from what women says to women actually means."},
            {"role": "user", "content": text_to_translate}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        translated_output = response.json()['choices'][0]['message']['content']
        return jsonify({"translatedText": translated_output})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except KeyError:
        return jsonify({"error": "Unexpected API response format"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)