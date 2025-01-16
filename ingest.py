# ingest.py

import os
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray

def read_text_file(file_path: str) -> str:
    """Read all text from a file and return as a single string."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def chunk_text(text: str, chunk_delimiter: str = "\n\n") -> List[str]:
    """Split text into a list of chunks or sections."""
    sections = text.split(chunk_delimiter)
    # Optionally filter out very short or empty chunks
    sections = [sec.strip() for sec in sections if sec.strip()]
    return sections

def build_faiss_index(sections: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> Tuple[faiss.IndexFlatL2, SentenceTransformer]:
    """
    Takes a list of text sections, computes embeddings, and returns a FAISS index
    along with the embedding model.
    """
    # Load embedding model
    embedding_model = SentenceTransformer(model_name)

    # Create embeddings
    embeddings = embedding_model.encode(sections, convert_to_tensor=True)
    embeddings_np: NDArray[np.float32] = embeddings.cpu().detach().numpy().astype(np.float32)

    # Initialize a simple FAISS index (L2 distance)
    d = embeddings_np.shape[1]
    faiss_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(d)

    # Add the embeddings
    if len(embeddings_np.shape) == 1:
        embeddings_np = embeddings_np.reshape(1, -1)

    faiss_index.add(embeddings_np)  # type: ignore
    print(f"FAISS index has {faiss_index.ntotal} items.")

    return faiss_index, embedding_model

def main() -> None:
    """
    This script reads the combined code from a text file,
    chunks it, and builds a FAISS index ready for queries.
    """
    # Path to your single text file
    file_path = os.path.join("data", "combined_code.txt")

    # 1. Read the text
    text_content = read_text_file(file_path)

    # 2. Chunk the text
    sections = chunk_text(text_content)

    # 3. Build FAISS index
    index, embedding_model = build_faiss_index(sections)

if __name__ == "__main__":
    main()
