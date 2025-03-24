# Updated app.py with authentication
from flask import send_from_directory
from flask import Flask, request, jsonify
from anonymization import anonymize_text
from llm_client import send_to_llm
import json
from flask_cors import CORS
from dotenv import load_dotenv
import os
import re
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
import bcrypt

load_dotenv()

app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour expiration

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

CORS(app)  # Add this as a fallback

CORS(app, resources={
    r"/*": {
        "origins": "*",  # Temporary wildcard for debugging
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"],
        "supports_credentials": True
    }
})

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(
            password.encode('utf-8'), 
            bcrypt.gensalt()
        ).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            self.password_hash.encode('utf-8')
        )


@app.route('/verify', methods=['GET'])
@jwt_required()
def verify_token():
    current_user = get_jwt_identity()
    return jsonify({"username": current_user}), 200


# Auth Routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400

    user = User(username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=username)
    #return jsonify(access_token=access_token), 200
    return jsonify({  # Add user info to response
        "access_token": access_token,
        "username": username
    }), 200

from flask import abort

@app.before_request
def check_json():
    if request.method in ['POST', 'PUT']:
        if not request.is_json:
            abort(415, "Request must be JSON")


@app.errorhandler(422)
def handle_unprocessable(err):
    return jsonify({
        "error": "Validation Error",
        "message": "Request data validation failed"
    }), 422

@app.errorhandler(400)
def handle_bad_request(err):
    return jsonify({
        "error": "Bad Request",
        "message": str(err)
    }), 400

@app.route('/process', methods=['POST'])
@jwt_required()
def process_request():
    try:
        # Add request validation
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        original_prompt = data.get("prompt")
        
        # Enhanced validation
        if not isinstance(original_prompt, str):
            return jsonify({"error": "Prompt must be a string"}), 422
            
        if len(original_prompt.strip()) < 1:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Add debug logging
        print(f"Processing prompt: {original_prompt}")
        print(f"Headers: {dict(request.headers)}")

        # ... rest of your existing code ...

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
        llm_response = send_to_llm(
            anonymized_prompt,
            placeholders=mapped_placeholders  # Pass placeholders for proper response handling
        )

        # ✅ Debug print before recontextualization
        print("\n📌 **LLM Response Before Cleaning:**\n", llm_response)

        # 🔹 Step 3: Recontextualization - Replace anonymized placeholders with original values
        for item in mapping:
            placeholder = re.escape(item["anonymized"])
            llm_response = re.sub(rf'{placeholder}', item["original"], llm_response)

        # ✅ Final cleanup of unnecessary placeholders
        llm_response = re.sub(r'<\w+_\d+>', '', llm_response)

        # ✅ Final debug print of cleaned response
        print("\n📌 **Final Response (After Cleaning):**\n", llm_response)

        # 🔹 Step 4: Return response
        return jsonify({
            "response": llm_response,
            "anonymized_prompt": anonymized_prompt,
            "mapping": mapping
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return jsonify({"status": "active"}), 200

@app.route('/<path:path>')
def catch_all(path):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint doesn't exist",
        "available_endpoints": ["/register", "/login", "/process"]
    }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
