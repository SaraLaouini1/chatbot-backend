# app.py

import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from anonymization import anonymize_text
from llm_client import send_to_llm

load_dotenv()

app = Flask(__name__)

# Allow your front‑end origins
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://chatbot-login.onrender.com",
            "http://localhost:5173"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

@app.route('/')
def health_check():
    return jsonify({"status": "active"}), 200

@app.route('/process', methods=['GET'])
def handle_get():
    return jsonify({"error": "Use POST method"}), 405

@app.route('/process', methods=['POST'])
def process_request():
    try:
        data = request.get_json(force=True)
        original_prompt = data.get("prompt", "")

        # 🔹 Step 1: Anonymize
        anonymized_prompt, mapping = anonymize_text(original_prompt)

        print("\n📌 Anonymized Prompt:\n", anonymized_prompt)
        print("\n📌 Mapping:\n", json.dumps(mapping, indent=2))

        # 🔹 Step 2: Send to LLM
        placeholders = [m["anonymized"] for m in mapping]
        llm_raw = send_to_llm(anonymized_prompt, placeholders)

        print("\n📌 LLM Raw Response:\n", llm_raw)

        # 🔹 Step 3: Re‑inject originals
        llm_recontext = llm_raw
        for m in mapping:
            esc = re.escape(m["anonymized"])
            llm_recontext = re.sub(rf'{esc}', m["original"], llm_recontext)

        # Remove any leftover tokens
        llm_final = re.sub(r'<\w+_\d+>', '', llm_recontext)
        print("\n📌 Final Response:\n", llm_final)

        return jsonify({
            "response": llm_final,
            "llm_raw": llm_raw,
            "llm_after_recontext": llm_recontext,
            "anonymized_prompt": anonymized_prompt,
            "mapping": mapping
        })

    except Exception as e:
        app.logger.error("Error in /process: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
