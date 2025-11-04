from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from together import Together
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) 

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

client = Together(api_key=TOGETHER_API_KEY)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text_to_translate = data.get('text', '')

    if not text_to_translate:
        return jsonify({"error": "No text provided"}), 400

    if not FINE_TUNED_MODEL:
        return jsonify({"error": "Fine-tuned model not configured"}), 500

    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a brutally honest relationship advisor for guys. "
                        "You speak like a bro giving tough love. "
                        "to understand the situation before giving advice. Never sugarcoat. "
                        "Always give EXACT phrases to say."
                    )
                },
                {"role": "user", "content": text_to_translate}
            ],
            max_tokens=100,
        )

        translated_output = response.choices[0].message.content.strip()
        return jsonify({"translatedText": translated_output})

    except Exception as e:
        return jsonify({"error": f"LLM request failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)