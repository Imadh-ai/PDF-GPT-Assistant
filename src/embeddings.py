import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Load model once globally (avoids reloading on every call)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> List[float]:
    """
    Get a single embedding vector using sentence-transformers.
    Runs locally — no API needed.

    Args:
        text: The text to embed

    Returns:
        A list of floats (the vector)
    """
    text = text.replace("\n", " ")
    vector = embedding_model.encode(text)
    return vector.tolist()


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Embed all chunks and attach the vector to each chunk dict.

    Args:
        chunks: List of chunk dicts from pdf_processor.py

    Returns:
        Same chunks but each now has an 'embedding' key
    """
    print(f"🔢 Embedding {len(chunks)} chunks...")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk["text"])
            chunk_with_embedding = {**chunk, "embedding": embedding}
            embedded_chunks.append(chunk_with_embedding)

            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"   ✅ Embedded {i + 1}/{len(chunks)} chunks")

        except Exception as e:
            print(f"   ⚠️  Skipping chunk {i} due to error: {e}")
            continue

    print(f"✅ Done! {len(embedded_chunks)} chunks embedded successfully.")
    return embedded_chunks