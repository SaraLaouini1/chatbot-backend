# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_migrate import Migrate  # Import Flask-Migrate
from dotenv import load_dotenv
import os, re, json

# Import your user model and database instance
from models import db, User
from anonymization import anonymize_text
from llm_client import send_to_llm

load_dotenv()

app = Flask(__name__, static_folder='../dist', static_url_path='')

# Configure app with environment variables
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate
CORS(app, resources={
    r"/process": {
        "origins": [
            "https://chatbot-login.onrender.com",
            "http://localhost:5173"
        ],
        "allow_headers": ["Authorization", "Content-Type"],
        "methods": ["GET", "POST", "PUT", "DELETE"]
    }
})

# No need for manual table creation; use migrations instead.

# Serve static files (for your frontend)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/'):  # Skip API routes
        return jsonify({"error": "Not found"}), 404

    static_file = os.path.join(app.static_folder, path)
    if os.path.exists(static_file) and path != "":
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "active"}), 200

# Registration endpoint
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    user = User.create_user(username, password)
    token = create_access_token(identity={"username": user.username, "role": user.role})
    return jsonify({"access_token": token}), 201

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        token = create_access_token(identity={"username": user.username, "role": user.role})
        return jsonify({"access_token": token})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# Protected process endpoint (example)
@app.route('/process', methods=['POST'])
@jwt_required()  # Require a valid JWT token
def process_request():
    try:
        data = request.json
        original_prompt = data.get("prompt", "")

        # Step 1: Anonymization
        anonymized_prompt, mapping = anonymize_text(original_prompt)
        print("\n📌 **Anonymized Prompt:**\n", anonymized_prompt)
        print("\n📌 **Mapping:**\n", json.dumps(mapping, indent=2))

        # Step 2: Send to LLM
        mapped_placeholders = [item["anonymized"] for item in mapping]
        llm_raw_response = send_to_llm(anonymized_prompt, placeholders=mapped_placeholders)
        print("\n📌 **LLM Response Before Cleaning:**\n", llm_raw_response)

        # Step 3: Recontextualization
        llm_after_recontext = llm_raw_response
        for item in mapping:
            placeholder = re.escape(item["anonymized"])
            llm_after_recontext = re.sub(rf'{placeholder}', item["original"], llm_after_recontext)
        llm_final_response = re.sub(r'<\w+_\d+>', '', llm_after_recontext)
        print("\n📌 **Final Response (After Cleaning):**\n", llm_final_response)

        # Step 4: Return response
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
