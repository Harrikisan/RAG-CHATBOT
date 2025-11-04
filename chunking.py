import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
 
load_dotenv()
 
def create_mini_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)
 
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
    "indices": [1,2],
    "summary": "Topic summary"
  }}
]
Chunks:
{formatted}
"""
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant", temperature=0.2)
    messages = [SystemMessage(content="Semantic text chunking assistant"), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content.strip()
 
def build_final_chunks(raw_chunks, llm_response):
    match = re.search(r"\[.*\]", llm_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in LLM response.")
    data = json.loads(match.group(0))
    final_chunks = []
    for group in data:
        indices = group["indices"]
        combined = " ".join([raw_chunks[i-1] for i in indices])
        final_chunks.append({"group": group["group"], "text": combined, "summary": group["summary"]})
    return final_chunks