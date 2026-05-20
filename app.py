from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

GEMINI_KEY = os.environ.get("GEMINI_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    system_instruction = data.get("system_instruction", "")
    contents = data.get("contents", [])

    body = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}
    }

    response = requests.post(GEMINI_URL, json=body)
    return jsonify(response.json()), response.status_code

if __name__ == "__main__":
    app.run(debug=False)