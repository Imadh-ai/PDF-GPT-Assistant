import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document excerpts.

Rules:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I couldn't find that information in the document"
- Be concise and accurate
- Quote relevant parts when helpful
- Never make up information
"""


def build_prompt(question: str, context: str) -> str:
    """
    Build the prompt combining context and question.

    Args:
        question: User's question
        context: Retrieved document excerpts

    Returns:
        Formatted prompt string
    """
    return f"""Here are the relevant excerpts from the document:

{context}

Based on the above excerpts, please answer this question:
{question}"""


def ask_question(
    question: str,
    context: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Send question + context to Groq LLaMA and get an answer.

    Args:
        question: User's question
        context: Retrieved document context
        chat_history: Previous Q&A pairs for multi-turn conversation

    Returns:
        LLM's answer as a string
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history for context if available
    if chat_history:
        for entry in chat_history[-3:]:  # Last 3 exchanges only
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})

    # Add current question with context
    messages.append({
        "role": "user",
        "content": build_prompt(question, context)
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,      # Low temp = more factual, less creative
        max_tokens=1024
    )

    return response.choices[0].message.content


def run_rag_pipeline(
    question: str,
    index,
    embedded_chunks: List[Dict],
    embedding_fn,
    chat_history: List[Dict] = None,
    top_k: int = 5
) -> Dict:
    """
    Full RAG pipeline: retrieve relevant chunks → ask LLM → return answer.

    Args:
        question: User's question
        index: FAISS index
        embedded_chunks: All embedded chunks
        embedding_fn: Function to embed query
        chat_history: Previous Q&A history
        top_k: Number of chunks to retrieve

    Returns:
        Dict with answer and source chunks
    """
    from src.retriever import search_similar_chunks, get_context_from_chunks

    # Step 1: Retrieve relevant chunks
    relevant_chunks = search_similar_chunks(
        question, index, embedded_chunks, embedding_fn, top_k
    )

    # Step 2: Build context from chunks
    context = get_context_from_chunks(relevant_chunks)

    # Step 3: Get answer from LLM
    answer = ask_question(question, context, chat_history)

    return {
        "question": question,
        "answer": answer,
        "source_chunks": relevant_chunks,
        "context_used": context
    }