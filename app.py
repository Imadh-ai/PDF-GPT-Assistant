import streamlit as st
import os
import time
from dotenv import load_dotenv
from src.pdf_processor import process_pdf
from src.embeddings import get_embedding, embed_chunks
from src.retriever import build_faiss_index
from src.llm_chain import run_rag_pipeline

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# ─── Session State Init ─────────────────────────────────────────────────────────
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ─── Helper Functions ───────────────────────────────────────────────────────────
def process_uploaded_pdf(uploaded_file):
    """Process PDF and build FAISS index."""
    with st.spinner("📄 Extracting text from PDF..."):
        chunks = process_pdf(uploaded_file)

    with st.spinner(f"🔢 Embedding {len(chunks)} chunks..."):
        embedded_chunks = embed_chunks(chunks)

    with st.spinner("🏗️ Building search index..."):
        index, embedded_chunks = build_faiss_index(embedded_chunks)

    return index, embedded_chunks


def export_transcript() -> str:
    """Export Q&A history as a formatted transcript."""
    if not st.session_state.chat_history:
        return ""

    lines = [
        f"PDF Q&A Transcript",
        f"Document: {st.session_state.pdf_name}",
        f"{'=' * 50}\n"
    ]

    for i, entry in enumerate(st.session_state.chat_history, 1):
        lines.append(f"Q{i}: {entry['question']}")
        lines.append(f"A{i}: {entry['answer']}")
        lines.append("-" * 40)

    return "\n".join(lines)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Q&A Assistant")
    st.markdown("Upload a PDF and ask questions about it!")
    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to start asking questions"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            # New PDF uploaded — reset everything
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.faiss_index = None
            st.session_state.embedded_chunks = None

        if not st.session_state.pdf_processed:
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                try:
                    index, embedded_chunks = process_uploaded_pdf(uploaded_file)
                    st.session_state.faiss_index = index
                    st.session_state.embedded_chunks = embedded_chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_processed = True
                    st.success("✅ PDF ready! Ask your questions.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
        else:
            st.success(f"✅ **{st.session_state.pdf_name}**")
            st.caption(f"📊 {len(st.session_state.embedded_chunks)} chunks indexed")

            if st.button("🔄 Upload New PDF", use_container_width=True):
                st.session_state.pdf_processed = False
                st.session_state.chat_history = []
                st.session_state.faiss_index = None
                st.session_state.embedded_chunks = None
                st.session_state.pdf_name = None
                st.rerun()

    st.divider()

    # Export transcript
    if st.session_state.chat_history:
        transcript = export_transcript()
        st.download_button(
            label="📥 Export Transcript",
            data=transcript,
            file_name=f"transcript_{st.session_state.pdf_name}.txt",
            mime="text/plain",
            use_container_width=True
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit + LLaMA 3.1 + FAISS")


# ─── Main Area ──────────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    # Welcome screen
    st.title("📄 PDF Q&A Assistant")
    st.markdown("### Welcome! Get started by uploading a PDF in the sidebar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1**\nUpload your PDF in the sidebar")
    with col2:
        st.info("🚀 **Step 2**\nClick 'Process PDF' to index it")
    with col3:
        st.info("💬 **Step 3**\nAsk any question about the document")

else:
    # Chat interface
    st.title(f"💬 Chatting with: {st.session_state.pdf_name}")
    st.divider()

    # Display chat history
    if not st.session_state.chat_history:
        st.info("💡 Ask your first question about the document below!")
    else:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])

    # Question input
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document and generating answer..."):
                try:
                    result = run_rag_pipeline(
                        question=question,
                        index=st.session_state.faiss_index,
                        embedded_chunks=st.session_state.embedded_chunks,
                        embedding_fn=get_embedding,
                        chat_history=st.session_state.chat_history
                    )

                    st.write(result["answer"])

                    # Show source excerpts in expander
                    with st.expander("📚 View Source Excerpts"):
                        for i, chunk in enumerate(result["source_chunks"][:3]):
                            st.markdown(f"**Excerpt {i+1}** (score: {chunk['similarity_score']:.3f})")
                            st.text(chunk["text"][:300] + "...")
                            st.divider()

                    # Save to history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"]
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")