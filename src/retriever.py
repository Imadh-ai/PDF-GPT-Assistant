import faiss
import numpy as np
from typing import List, Dict


def build_faiss_index(embedded_chunks: List[Dict]) -> tuple:
    """
    Build a FAISS index from embedded chunks for fast similarity search.

    Args:
        embedded_chunks: List of chunk dicts with 'embedding' key

    Returns:
        (faiss_index, chunks) tuple
    """
    print("🏗️  Building FAISS index...")

    # Extract embeddings into a numpy matrix
    embeddings = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float32
    )

    # Get vector dimension
    dimension = embeddings.shape[1]

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index (Inner Product = cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors of dimension {dimension}")
    return index, embedded_chunks


def search_similar_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    embedded_chunks: List[Dict],
    embedding_fn,
    top_k: int = 5
) -> List[Dict]:
    """
    Search for the most relevant chunks for a given query.

    Args:
        query: The user's question
        index: FAISS index
        embedded_chunks: Original chunks with metadata
        embedding_fn: Function to embed the query
        top_k: Number of top results to return

    Returns:
        List of most relevant chunks with similarity scores
    """
    # Embed the query
    query_vector = np.array([embedding_fn(query)], dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(query_vector)

    # Search the index
    scores, indices = index.search(query_vector, top_k)

    # Collect results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:  # -1 means no result found
            chunk = embedded_chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

    return results


def get_context_from_chunks(chunks: List[Dict]) -> str:
    """
    Combine retrieved chunks into a single context string for the LLM.

    Args:
        chunks: List of retrieved chunks

    Returns:
        Combined context string
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Excerpt {i+1}]\n{chunk['text']}")

    return "\n\n".join(context_parts)