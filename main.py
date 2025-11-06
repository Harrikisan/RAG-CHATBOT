import os
import json
import re
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
# -----------------------------
# Setup
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")
 
app = FastAPI(title="Docling + Agentic Chunking RAG API")
 
# -----------------------------
# Initialize Groq LLM
# -----------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.2
)
 
# -----------------------------
# In-memory FAISS store
# -----------------------------
vector_store = None
 
 
# -----------------------------
# Utility: Agentic grouping (LLM-based)
# -----------------------------
def group_chunks_with_llm(chunks: List[str]) -> str:
    formatted = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""
You are an expert in semantic chunking.
Below are small text chunks labeled with numbers starting from 1.
 
Task:
- Group related chunks by topic.
- Ensure indices are within range 1 to {len(chunks)}.
- Return only valid JSON format like:
[
  {{
    "group": 1,
    "indices": [1, 2],
    "summary": "Topic summary here"
  }}
]
 
No extra text, only JSON.
Chunks:
{formatted}
"""
    messages = [
        SystemMessage(content="You are a semantic text chunking assistant."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return response.content.strip()
 
 
def build_final_chunks(raw_chunks: List[str], llm_response: str):
    # Extract JSON from response
    match = re.search(r"\[.*\]", llm_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in LLM response.")
    data = json.loads(match.group(0))
    final_chunks = []
    total_chunks = len(raw_chunks)
 
    for group in data:
        valid_indices = [i for i in group["indices"] if 1 <= i <= total_chunks]
        if not valid_indices:
            continue  # skip invalid or empty group
 
        combined_text = " ".join([raw_chunks[i - 1] for i in valid_indices])
        final_chunks.append({
            "group": group["group"],
            "text": combined_text,
            "summary": group.get("summary", "")
        })
 
    if not final_chunks:
        raise ValueError("No valid groups were produced by LLM.")
 
    return final_chunks 
# -----------------------------
# Endpoint: Query RAG (Hybrid mode)
# -----------------------------
@app.post("/query")
async def query_rag(question: str):
    try:
        global vector_store
 
        logger.info(f"Received query: {question}")
 
        # ----------------------------
        # Case 1: No documents uploaded yet → pure LLM
        # ----------------------------
        if not vector_store:
            logger.info("No documents uploaded. Using pure LLM mode.")
            response = llm.invoke([
                SystemMessage(content="You are a general-purpose assistant."),
                HumanMessage(content=question)
            ])
            return JSONResponse({"mode": "llm-only", "answer": response.content})
 
        # ----------------------------
        # Case 2: Retrieve context from FAISS
        # ----------------------------
        results = vector_store.similarity_search(question, k=3)
        if not results or all(len(r.page_content.strip()) == 0 for r in results):
            logger.info("No relevant context found. Falling back to pure LLM.")
            response = llm.invoke([
                SystemMessage(content="You are a general-purpose assistant."),
                HumanMessage(content=question)
            ])
            return JSONResponse({"mode": "llm-fallback", "answer": response.content})
 
        # ----------------------------
        # Case 3: Combine RAG + LLM
        # ----------------------------
        context = "\n".join([r.page_content for r in results])
 
        prompt = f"""
You are an expert AI assistant.
Answer the question using the following context from the uploaded documents.
If the answer cannot be found in the context, clearly mention that you are answering
based on your general knowledge.
 
Context:
{context}
 
Question: {question}
Answer:
"""
        response = llm.invoke([
            SystemMessage(content="You are a helpful RAG assistant."),
            HumanMessage(content=prompt)
        ])
 
        return JSONResponse({
            "mode": "rag",
            "answer": response.content,
            "context_used": context
        })
 
    except Exception as e:
        logger.exception("Error during query process")
        return JSONResponse({"status": "error", "detail": str(e)})
# -----------------------------
# Endpoint: Query RAG
# -----------------------------
@app.post("/query")
async def query_rag(question: str):
    try:
        global vector_store
        if not vector_store:
            return JSONResponse({"status": "error", "detail": "No documents uploaded yet."})
 
        logger.info(f"Received query: {question}")
 
        # Retrieve top 3 similar chunks
        results = vector_store.similarity_search(question, k=3)
        context = "\n".join([r.page_content for r in results])
 
        prompt = f"""
You are an expert AI assistant. Use the context below to answer the question.
 
Context:
{context}
 
Question: {question}
Answer:
"""
        response = llm.invoke([
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=prompt)
        ])
 
        return JSONResponse({"answer": response.content, "context": context})
 
    except Exception as e:
        logger.exception("Error during query process")
        return JSONResponse({"status": "error", "detail": str(e)})
 
 
# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "RAG API running successfully"}