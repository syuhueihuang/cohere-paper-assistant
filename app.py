import os

import cohere
import streamlit as st
from dotenv import load_dotenv

from utils import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    answer_with_rag,
    build_vector_index,
    explain_concept,
    extract_text_from_pdf,
    summarize_paper,
)


load_dotenv()

st.set_page_config(
    page_title="Research Paper QA with Cohere",
    page_icon="📄",
    layout="wide",
)


def get_client(api_key: str) -> cohere.ClientV2:
    return cohere.ClientV2(api_key=api_key)


def get_default_api_key() -> str:
    env_key = os.getenv("COHERE_API_KEY", "")
    try:
        return st.secrets.get("COHERE_API_KEY", env_key)
    except Exception:
        return env_key


if "paper_name" not in st.session_state:
    st.session_state.paper_name = None
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Cohere API Key",
        value=get_default_api_key(),
        type="password",
        help="Use Streamlit secrets or a local .env file for deployment.",
    )
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embed model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Source chunks", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk size", min_value=700, max_value=1800, value=1100, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=100, max_value=400, value=180, step=20)


st.title("Research Paper QA with Cohere")
st.caption("Upload a research paper, search it with Cohere embeddings, and ask grounded questions with source chunks.")

uploaded_pdf = st.file_uploader("Upload a research paper PDF", type=["pdf"])

if uploaded_pdf:
    is_new_file = uploaded_pdf.name != st.session_state.paper_name

    if is_new_file:
        st.session_state.paper_name = uploaded_pdf.name
        st.session_state.paper_text = ""
        st.session_state.chunks = []
        st.session_state.index = None

    if st.button("Process paper", type="primary"):
        if not api_key:
            st.error("Please enter your Cohere API key first.")
            st.stop()

        with st.spinner("Reading PDF and building semantic index..."):
            paper_text, page_texts = extract_text_from_pdf(uploaded_pdf.getvalue())
            client = get_client(api_key)
            chunks, index = build_vector_index(
                client=client,
                page_texts=page_texts,
                embed_model=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.paper_text = paper_text
            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success(f"Indexed {len(st.session_state.chunks)} chunks from {uploaded_pdf.name}.")
    elif not st.session_state.chunks:
        st.info("Click Process paper to extract text and build the FAISS index.")


if st.session_state.chunks:
    left, right = st.columns([0.68, 0.32], gap="large")

    with right:
        st.subheader("Paper")
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Characters", f"{len(st.session_state.paper_text):,}")

        with st.expander("Preview extracted text"):
            st.text(st.session_state.paper_text[:3000])

    with left:
        tabs = st.tabs(["Ask", "Summarize", "Explain", "Search"])

        with tabs[0]:
            question = st.text_input(
                "Ask a question about the paper",
                placeholder="What is the main contribution of this paper?",
            )
            if st.button("Generate answer", type="primary", disabled=not question):
                if not api_key:
                    st.error("Please enter your Cohere API key first.")
                    st.stop()

                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    client = get_client(api_key)
                    result = answer_with_rag(
                        client=client,
                        question=question,
                        chunks=st.session_state.chunks,
                        index=st.session_state.index,
                        chat_model=chat_model,
                        embed_model=embed_model,
                        top_k=top_k,
                    )

                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Source Chunks")
                for source in result["sources"]:
                    with st.expander(
                        f"Chunk {source['chunk_id']} · page {source['page']} · score {source['score']:.3f}"
                    ):
                        st.write(source["text"])

        with tabs[1]:
            summary_style = st.selectbox(
                "Summary type",
                ["Executive summary", "Technical summary", "Interview talking points"],
            )
            if st.button("Summarize paper"):
                if not api_key:
                    st.error("Please enter your Cohere API key first.")
                    st.stop()

                with st.spinner("Generating summary..."):
                    client = get_client(api_key)
                    summary = summarize_paper(
                        client=client,
                        chunks=st.session_state.chunks,
                        chat_model=chat_model,
                        style=summary_style,
                    )
                st.write(summary)

        with tabs[2]:
            concept = st.text_input(
                "Concept to explain",
                placeholder="e.g., contrastive learning, diffusion prior, feature pyramid",
            )
            if st.button("Explain concept", disabled=not concept):
                if not api_key:
                    st.error("Please enter your Cohere API key first.")
                    st.stop()

                with st.spinner("Finding context and explaining..."):
                    client = get_client(api_key)
                    result = explain_concept(
                        client=client,
                        concept=concept,
                        chunks=st.session_state.chunks,
                        index=st.session_state.index,
                        chat_model=chat_model,
                        embed_model=embed_model,
                        top_k=top_k,
                    )

                st.subheader("Explanation")
                st.write(result["answer"])
                st.subheader("Related Source Chunks")
                for source in result["sources"]:
                    with st.expander(
                        f"Chunk {source['chunk_id']} · page {source['page']} · score {source['score']:.3f}"
                    ):
                        st.write(source["text"])

        with tabs[3]:
            search_query = st.text_input(
                "Semantic search",
                placeholder="Find sections about experiments, limitations, or architecture.",
            )
            if st.button("Find sections", disabled=not search_query):
                if not api_key:
                    st.error("Please enter your Cohere API key first.")
                    st.stop()

                with st.spinner("Searching paper..."):
                    client = get_client(api_key)
                    result = answer_with_rag(
                        client=client,
                        question=f"Return the most relevant sections for: {search_query}",
                        chunks=st.session_state.chunks,
                        index=st.session_state.index,
                        chat_model=chat_model,
                        embed_model=embed_model,
                        top_k=top_k,
                        search_only=True,
                    )

                for source in result["sources"]:
                    with st.expander(
                        f"Chunk {source['chunk_id']} · page {source['page']} · score {source['score']:.3f}"
                    ):
                        st.write(source["text"])
else:
    st.info("Upload a PDF and click Process paper to start.")
