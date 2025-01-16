# app.py

import os
from flask import Flask, request, jsonify
from ingest import read_text_file, chunk_text, build_faiss_index
from chat import Chatbot

app = Flask(__name__)

# --- Setup on app startup ---
file_path = os.path.join("data", "combined_code.txt")
text_content = read_text_file(file_path)
sections = chunk_text(text_content)
faiss_index, embedding_model = build_faiss_index(sections)

# Initialize the Chatbot with HF model = "gpt2" (change to your preferred model)
chatbot = Chatbot(faiss_index, embedding_model, sections, hf_model="gpt2")

# --- Routes ---

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No 'query' provided"}), 400

    response = chatbot.generate_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
