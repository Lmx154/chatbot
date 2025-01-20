# Code Context Chatbot

A Flask-based chatbot that uses FAISS for semantic search and Hugging Face models to generate contextual responses about code repositories.

## Features

- Semantic search using FAISS indexing
- Context-aware responses using Hugging Face transformers
- RESTful API endpoint for chat interactions
- Support for processing and understanding code documentation
- Environment variable configuration for API tokens

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install python-dotenv transformers torch sentence-transformers faiss-cpu flask
```

4. Create a `.env` file in the root directory and add your Hugging Face API token:
```
hugging_chatbot_read_token=your_token_here
```

## Project Structure

```
chatbot/
├── app.py              # Flask application entry point
├── chat.py            # Chatbot implementation
├── ingest.py          # Text processing and FAISS indexing
├── data/              # Directory for source data
│   └── combined_code.txt  # Source code and documentation
└── .env              # Environment variables
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. The server will start on `http://localhost:5000`

3. Send chat requests to the `/chat` endpoint:
```bash
curl -X POST http://localhost:5000/chat \
-H "Content-Type: application/json" \
-d '{"query":"What is this project about?"}'
```

## API Endpoints

### POST /chat
Send a chat message and receive a response.

**Request Body:**
```json
{
    "query": "Your question here"
}
```

**Response:**
```json
{
    "response": "Generated response based on context"
}
```

## How It Works

1. **Text Ingestion:**
   - Source code and documentation are read from `data/combined_code.txt`
   - Text is chunked into manageable sections
   - FAISS index is built using sentence embeddings

2. **Query Processing:**
   - User queries are converted to embeddings
   - FAISS finds the most relevant text chunks
   - Context is assembled from matching chunks

3. **Response Generation:**
   - Context and query are formatted into a prompt
   - Hugging Face model generates a contextual response

## Configuration

The following environment variables can be configured in `.env`:

- `hugging_chatbot_read_token`: Your Hugging Face API token

## Limitations

- Responses are limited by the context window size of the model
- Quality depends on the source material in combined_code.txt
- Rate limits may apply based on your Hugging Face API plan

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FAISS by Facebook Research
- Hugging Face Transformers
- Sentence Transformers
- Flask Framework
