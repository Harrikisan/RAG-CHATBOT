from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json, os, re
 
# Load GROQ API key
load_dotenv()
 
# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",  # You can also use "llama3-70b" or "llama3-70b-versatile"
    temperature=0.2
)
 
# Step 1: Create mini chunks
def create_mini_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return text_splitter.split_text(text)
 
# Example text
text = """Artificial intelligence (AI) is transforming the world.
Large language models are part of that revolution.
They help with tasks like summarization, reasoning, and creative writing.
AI also supports industries from healthcare to agriculture.
The technology continues to evolve rapidly with new breakthroughs each year."""
 
mini_chunks = create_mini_chunks(text)
print("\n--- Mini Chunks ---")
print(mini_chunks)
 
# Step 2: Group related chunks using Groq
def group_chunks_with_llm(chunks):
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
 
 
def build_final_chunks(raw_text_chunks, llm_response):
    # Extract only JSON from model output (even if LLM adds explanation)
    match = re.search(r"\[.*\]", llm_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in LLM response.")
 
    json_text = match.group(0)
    data = json.loads(json_text)
 
    final_chunks = []
    for group in data:
        indices = group["indices"]
        combined = " ".join([raw_text_chunks[i - 1] for i in indices])
        final_chunks.append({
            "group": group["group"],
            "text": combined,
            "summary": group["summary"]
        })
    return final_chunks 
grouped = group_chunks_with_llm(mini_chunks)
print("\n--- LLM Grouping ---")
print(grouped)
 
# Step 3: Combine grouped chunks
def build_final_chunks(raw_chunks, llm_response):
    data = json.loads(llm_response)
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
 
final_chunks = build_final_chunks(mini_chunks, grouped)
print("\n--- Final Combined Chunks ---")
for chunk in final_chunks:
    print(f"Group {chunk['group']}: {chunk['summary']}")
 
# Step 4: Embed with Hugging Face model and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [chunk["text"] for chunk in final_chunks]
metadatas = [{"summary": chunk["summary"]} for chunk in final_chunks]
 
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
print("\n✅ FAISS vector store created successfully!")