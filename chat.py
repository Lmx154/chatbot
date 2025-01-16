# chat.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Tuple, Dict, Any

class Chatbot:
    def __init__(self, faiss_index: faiss.IndexFlatL2,
                 embedding_model: SentenceTransformer,
                 sections: List[str],
                 hf_model: str = "gpt2"):
        """
        Initialize the chatbot with a FAISS index, an embedding model, the raw text sections,
        and a Hugging Face model (defaults to 'gpt2').
        """
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.sections = sections
        self.generator = pipeline("text-generation", model=hf_model)

    def search_faiss(self, query: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Convert user query into an embedding, then search the FAISS index for top-k results."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_embedding_np = np.array(query_embedding.cpu().detach().numpy(), dtype='float32')

        distances, indices = self.faiss_index.search(query_embedding_np, k)
        return indices[0], distances[0]

    def generate_response(self, query: str) -> str:
        """Use the top matching chunks as context, then generate a response using HF pipeline."""
        # 1. Find relevant chunks
        top_indices, _ = self.search_faiss(query, k=3)
        context = "\n".join(self.sections[int(idx)] for idx in top_indices)

        # 2. Build a prompt that includes the context
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # 3. Generate text using the HF pipeline
        result = self.generator(prompt, max_length=256, do_sample=True, top_p=0.9)

        # Handle the generator output correctly
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get('generated_text', '')
            return str(result[0])

        return "I couldn't generate a response."
