# chat.py

import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Tuple

class Chatbot:
    def __init__(self, faiss_index: faiss.IndexFlatL2,
                 embedding_model: SentenceTransformer,
                 sections: List[str],
                 hf_model: str = "gpt2"):
        """
        Initialize the chatbot with a FAISS index, embedding model, sections,
        and Hugging Face model.
        """
        # Load environment variables
        load_dotenv()
        self.hf_token = os.getenv('hugging_chatbot_read_token')

        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.sections = sections

        # Initialize the model with authentication
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model, token=self.hf_token)
        self.generator = pipeline("text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer)

    def search_faiss(self, query: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Convert user query into an embedding, then search the FAISS index."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_embedding_np = np.array(query_embedding.cpu().detach().numpy(), dtype='float32')

        distances, indices = self.faiss_index.search(query_embedding_np, k)
        return indices[0], distances[0]

    def generate_response(self, query: str) -> str:
        """Generate a response using relevant context and the HF model."""
        # Find relevant chunks
        top_indices, _ = self.search_faiss(query, k=3)
        context = "\n".join(self.sections[int(idx)] for idx in top_indices)

        # Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Generate response
        try:
            result = self.generator(prompt,
                                  max_length=256,
                                  do_sample=True,
                                  top_p=0.9,
                                  num_return_sequences=1)

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return result[0].get('generated_text', '').replace(prompt, '').strip()
                return str(result[0]).replace(prompt, '').strip()

            return "I couldn't generate a response."

        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response."
