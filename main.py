import os
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List
from langchain_docling import DoclingLoader  # no ExportType needed
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
# Initialize FastAPI
app = FastAPI(title="Docling + Agentic Chunking RAG API")
 
# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.2
)
 
# In-memory FAISS store (can also persist to disk)
vector_store = None
 
# -----------------------------
# Utility: Agentic grouping (optional)
# -----------------------------
def group_chunks_with_llm(chunks: List[str]) -> str:
    formatted = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""
You are an expert in semantic chunking.
Below are small text chunks labeled with numbers.
 
Task:
- Group related chunks by topic.
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
    for group in data:
        indices = group["indices"]
        combined_text = " ".join([raw_chunks[i-1] for i in indices])
        final_chunks.append({
            "group": group["group"],
            "text": combined_text,
            "summary": group["summary"]
        })
    return final_chunks
 
# -----------------------------
# Endpoint: Upload document
# -----------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
 
        # Load document via Docling
        loader = DoclingLoader(file_path=temp_path)
        docs = loader.load()
        os.remove(temp_path)  # delete temp file
 
        # Extract chunks
        chunks = [d.page_content for d in docs]
 
        # Agentic grouping with Groq LLM
        grouped_json = group_chunks_with_llm(chunks)
        final_chunks = build_final_chunks(chunks, grouped_json)
 
        # Embed chunks with HuggingFace and store in FAISS
        global vector_store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texts = [c["text"] for c in final_chunks]
        metadatas = [{"summary": c["summary"]} for c in final_chunks]
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
 
        return JSONResponse({"status": "success", "num_chunks": len(final_chunks)})
 
    except Exception as e:
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
        return JSONResponse({"status": "error", "detail": str(e)})