import pdfplumber
import PyPDF2
import re
from typing import List, Dict


def extract_text_from_pdf(file) -> str:
    """
    Extract raw text from an uploaded PDF file.
    Tries pdfplumber first (better accuracy), falls back to PyPDF2.
    """
    text = ""

    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        # Fallback to PyPDF2
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove excessive whitespace
    - Remove weird characters
    - Normalize line breaks
    """
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks for better RAG retrieval.

    Args:
        text: The full extracted PDF text
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of dicts with chunk text and metadata
    """
    words = text.split()
    chunks = []
    chunk_index = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_index,
            "text": chunk_text_str,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": min(end, len(words))
        })

        chunk_index += 1
        start += chunk_size - overlap  # Move forward with overlap

    return chunks


def process_pdf(file) -> List[Dict]:
    """
    Full pipeline: extract → clean → chunk

    Args:
        file: Uploaded PDF file object (from Streamlit)

    Returns:
        List of text chunks with metadata
    """
    print("📄 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(file)

    if not raw_text:
        raise ValueError("Could not extract any text from this PDF. It may be scanned/image-based.")

    print("🧹 Cleaning text...")
    cleaned = clean_text(raw_text)

    print("✂️  Chunking text...")
    chunks = chunk_text(cleaned)

    print(f"✅ Done! Created {len(chunks)} chunks from {len(cleaned.split())} words.")
    return chunks