from flask import Flask, request, jsonify
from anonymization import anonymize_text
from llm_client import send_to_llm
import json
from flask_cors import CORS
from dotenv import load_dotenv
import os
import re
from flask_limiter import Limiter

load_dotenv()

app = Flask(__name__)

# Rate limiting for security
limiter = Limiter(
    app=app,
    key_func=lambda: request.headers.get('X-Real-IP', request.remote_addr)
)

# Render-optimized CORS
CORS(app, resources={
    r"/process": {
        "origins": [
            "https://your-frontend-name.onrender.com",
            "https://chatbot-login.onrender.com"
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
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
@limiter.limit("200/hour")
def process_request():
    try:
        data = request.json
        original_prompt = data.get("prompt", "")

        anonymized_prompt, mapping = anonymize_text(original_prompt)
        
        print("\n📌 **Anonymized Prompt:**\n", anonymized_prompt)
        print("\n📌 **Mapping:**\n", json.dumps(mapping, indent=2))

        mapped_placeholders = [item["anonymized"] for item in mapping]
        llm_raw_response = send_to_llm(anonymized_prompt, mapped_placeholders)

        print("\n📌 **LLM Response Before Cleaning:**\n", llm_raw_response)

        llm_after_recontext = llm_raw_response
        for item in mapping:
            placeholder = re.escape(item["anonymized"])
            llm_after_recontext = re.sub(rf'{placeholder}', item["original"], llm_after_recontext)

        llm_final_response = re.sub(r'<\w+_\d+>', '', llm_after_recontext)
        print("\n📌 **Final Response (After Cleaning):**\n", llm_final_response)

        return jsonify({
            "response": llm_final_response,
            "llm_raw": llm_raw_response,
            "llm_after_recontext": llm_after_recontext,
            "anonymized_prompt": anonymized_prompt,
            "mapping": mapping
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
