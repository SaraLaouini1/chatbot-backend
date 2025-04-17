
from flask import Flask, request, jsonify
from anonymization import anonymize_text
from llm_client import send_to_llm
from flask import Flask, request, jsonify
import json

from flask_cors import CORS
from dotenv import load_dotenv
import os
import re

import time
from collections import defaultdict
from datetime import datetime

from anonymization import LegalAnonymizer

legal_anonymizer = LegalAnonymizer()

load_dotenv()

app = Flask(__name__)




# Update CORS configuration at the top of app.py
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
        data = request.json
        original_prompt = data.get("prompt", "")

        # 🔹 Step 1: Anonymization
        #anonymized_prompt, mapping = anonymize_text(original_prompt)
        anonymized_prompt, mapping = legal_anonymizer.anonymize(original_prompt)


        # Validate placeholders in the anonymized prompt
        #expected_placeholders = {item["anonymized"] for item in mapping}
        #found_placeholders = set(re.findall(r"<\w+_\d+>", anonymized_prompt))
        
        #if expected_placeholders != found_placeholders:
            #raise ValueError(f"Placeholder mismatch. Expected: {expected_placeholders}, Found: {found_placeholders}")

        # ✅ Better debug prints for the anonymized prompt and mapping
        print("\n📌 **Anonymized Prompt:**\n", anonymized_prompt)
        print("\n📌 **Mapping:**\n", json.dumps(mapping, indent=2))

        # 🔹 Step 2: Send to LLM
        mapped_placeholders = [item["anonymized"] for item in mapping]
        llm_raw_response  = send_to_llm(
            anonymized_prompt,
            placeholders=mapped_placeholders
        )

        # ✅ Debug print before recontextualization
        print("\n📌 **LLM Response Before Cleaning:**\n", llm_raw_response )

        # 🔹 Step 3: Recontextualization - Replace anonymized placeholders with original values
        llm_after_recontext = llm_raw_response
        for item in mapping:
            placeholder = re.escape(item["anonymized"])
            llm_after_recontext = re.sub(rf'{placeholder}', item["original"], llm_after_recontext)

        # ✅ Final cleanup of unnecessary placeholders
        llm_final_response  = re.sub(r'<\w+_\d+>', '', llm_after_recontext)

        # ✅ Final debug print of cleaned response
        print("\n📌 **Final Response (After Cleaning):**\n", llm_final_response)



        # 🔹 Step 4: Return response
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
