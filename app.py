
from flask import Flask, request, jsonify
from anonymization import anonymize_text
from llm_client import send_to_llm
import json

from flask_cors import CORS
from dotenv import load_dotenv
import os
import re

from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash


load_dotenv()

app = Flask(__name__)

# Add after app creation
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL').replace("postgres://", "postgresql://", 1)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', os.urandom(32).hex())
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour expiration

db = SQLAlchemy(app)
jwt = JWTManager(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(120))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username exists"}), 400
        
    user = User(
        username=data['username'],
        password_hash=generate_password_hash(data['password']),
        is_active=True
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User created"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.is_active or not check_password_hash(user.password_hash, data['password']):
        return jsonify({"error": "Invalid credentials"}), 401
        
    return jsonify(access_token=create_access_token(identity=user.id))

@app.route('/admin/ban/<username>', methods=['POST'])
@jwt_required()
def ban_user(username):
    current_user = User.query.get(get_jwt_identity())
    if not current_user.is_admin:
        return jsonify({"error": "Admin required"}), 403
    
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    user.is_active = False
    db.session.commit()
    return jsonify({"message": f"{username} banned"})

# Update CORS config
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://chatbot-login.onrender.com",
            "http://localhost:5173"
        ],
        "methods": ["*"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Add before request check
@app.before_request
def check_active():
    if request.endpoint in ['login', 'register', 'health_check']:
        return
    current_user = User.query.get(get_jwt_identity())
    if not current_user or not current_user.is_active:
        return jsonify({"error": "Account disabled"}), 403

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
@jwt_required()
def process_request():
    try:
        data = request.json
        original_prompt = data.get("prompt", "")

        # 🔹 Step 1: Anonymization
        anonymized_prompt, mapping = anonymize_text(original_prompt)

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
            placeholders=mapped_placeholders  # Pass placeholders for proper response handling
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
