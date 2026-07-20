from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import cohere
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DEFAULT_CHAT_MODEL = "command-a-03-2025"
DEFAULT_EMBED_MODEL = "embed-v4.0"


@dataclass
class PaperChunk:
    chunk_id: int
    page: int
    text: str


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    page_texts = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        clean_text = " ".join(text.split())
        if clean_text:
            page_texts.append({"page": page_idx, "text": clean_text})

    if not page_texts:
        raise ValueError("No extractable text found in the PDF.")

    full_text = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in page_texts)
    return full_text, page_texts


def make_chunks(
    page_texts: list[dict[str, Any]],
    chunk_size: int = 1100,
    chunk_overlap: int = 180,
) -> list[PaperChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 1
    for page in page_texts:
        for text in splitter.split_text(page["text"]):
            chunks.append(PaperChunk(chunk_id=chunk_id, page=page["page"], text=text))
            chunk_id += 1

    return chunks


def _embed_texts(
    client: cohere.ClientV2,
    texts: list[str],
    model: str,
    input_type: str,
    batch_size: int = 64,
) -> np.ndarray:
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embed(
            model=model,
            texts=batch,
            input_type=input_type,
            embedding_types=["float"],
        )
        embeddings.extend(response.embeddings.float)

    matrix = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_vector_index(
    client: cohere.ClientV2,
    page_texts: list[dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = 1100,
    chunk_overlap: int = 180,
) -> tuple[list[PaperChunk], faiss.IndexFlatIP]:
    chunks = make_chunks(page_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    doc_embeddings = _embed_texts(
        client=client,
        texts=[chunk.text for chunk in chunks],
        model=embed_model,
        input_type="search_document",
    )

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return chunks, index


def retrieve_chunks(
    client: cohere.ClientV2,
    query: str,
    chunks: list[PaperChunk],
    index: faiss.IndexFlatIP,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    query_embedding = _embed_texts(
        client=client,
        texts=[query],
        model=embed_model,
        input_type="search_query",
    )
    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    sources = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx == -1:
            continue
        chunk = chunks[int(idx)]
        sources.append(
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "text": chunk.text,
                "score": float(score),
            }
        )
    return sources


def _chat(
    client: cohere.ClientV2,
    prompt: str,
    documents: list[dict[str, Any]] | None,
    chat_model: str,
) -> str:
    response = client.chat(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful AI research paper assistant. Answer only from the supplied paper "
                    "context when context is provided. If the paper does not contain the answer, say so clearly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        documents=documents,
    )
    return response.message.content[0].text


def answer_with_rag(
    client: cohere.ClientV2,
    question: str,
    chunks: list[PaperChunk],
    index: faiss.IndexFlatIP,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 4,
    search_only: bool = False,
) -> dict[str, Any]:
    sources = retrieve_chunks(
        client=client,
        query=question,
        chunks=chunks,
        index=index,
        embed_model=embed_model,
        top_k=top_k,
    )

    if search_only:
        return {"answer": "", "sources": sources}

    documents = [
        {
            "id": f"chunk-{source['chunk_id']}",
            "data": {
                "text": source["text"],
                "page": str(source["page"]),
                "chunk_id": str(source["chunk_id"]),
            },
        }
        for source in sources
    ]

    prompt = f"""
Question:
{question}

Instructions:
- Give a concise but complete answer.
- Cite page and chunk numbers when useful.
- Separate assumptions from facts stated in the paper.
"""
    answer = _chat(client=client, prompt=prompt, documents=documents, chat_model=chat_model)
    return {"answer": answer, "sources": sources}


def summarize_paper(
    client: cohere.ClientV2,
    chunks: list[PaperChunk],
    chat_model: str = DEFAULT_CHAT_MODEL,
    style: str = "Executive summary",
    max_chunks: int = 12,
) -> str:
    sampled_chunks = chunks[:max_chunks]
    documents = [
        {
            "id": f"chunk-{chunk.chunk_id}",
            "data": {
                "text": chunk.text,
                "page": str(chunk.page),
                "chunk_id": str(chunk.chunk_id),
            },
        }
        for chunk in sampled_chunks
    ]

    prompt = f"""
Create a {style.lower()} of this research paper.

Include:
1. Problem
2. Proposed method
3. Main contributions
4. Experiments and datasets
5. Limitations or open questions
"""
    return _chat(client=client, prompt=prompt, documents=documents, chat_model=chat_model)


def explain_concept(
    client: cohere.ClientV2,
    concept: str,
    chunks: list[PaperChunk],
    index: faiss.IndexFlatIP,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 4,
) -> dict[str, Any]:
    result = answer_with_rag(
        client=client,
        question=f"Explain this concept from the paper: {concept}",
        chunks=chunks,
        index=index,
        chat_model=chat_model,
        embed_model=embed_model,
        top_k=top_k,
    )
    return result
