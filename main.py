from fastapi import FastAPI, UploadFile, File
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunking import create_mini_chunks, group_chunks_with_llm, build_final_chunks
from dotenv import load_dotenv
import shutil, os
 
load_dotenv()
 
app = FastAPI(title="Local RAG API")
 
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
# Load existing FAISS DB if available
if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings)
else:
    db = None
 

# Upload endpoint
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
 
    # Read text (txt only for simplicity; PDFs can be added later)
    text = open(file_path, "r", encoding="utf-8").read()
 
    # Chunk & group
    mini_chunks = create_mini_chunks(text)
    grouped = group_chunks_with_llm(mini_chunks)
    final_chunks = build_final_chunks(mini_chunks, grouped)
 
    # Embed & store
    global db
    texts = [c["text"] for c in final_chunks]
    metadatas = [{"summary": c["summary"]} for c in final_chunks]
 
    if db:
        db.add_texts(texts, metadatas=metadatas)
    else:
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
 
    db.save_local("faiss_index")
    return {"status": "success", "chunks_added": len(final_chunks)}
 
# Query endpoint
@app.post("/query")
async def query_document(query: str):
    if not db:
        return {"error": "FAISS index not initialized."}
 
    results = db.similarity_search(query, k=3)
    context = "\n".join([res.page_content for res in results])
 
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    import os
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant", temperature=0.2)
 
    prompt = f"""
You are a helpful AI assistant. Answer the question based on the context below:
 
Context:
{context}
 
Question: {query}
Answer:
"""
    response = llm.invoke([
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=prompt)
    ])
    return {"answer": response.content}