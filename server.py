import os
import sys
import time
import json
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from google import genai
from google.genai import types

# LlamaIndex Imports
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import qdrant_client
from qdrant_client.http import models

# --- 1. Configuration & Setup ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
PERSIST_DIR = "/app/storage_metadata"
REGISTRY_FILE = "/app/storage_metadata/ingested_registry.json" # 👈 ไฟล์สมุดเช็คชื่อ
COLLECTION_NAME = "research_memory_final"

GLOBAL_INDEX = None
STORAGE_CONTEXT = None

# --- Helper: จัดการสมุดเช็คชื่อ ---
def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_registry(processed_files):
    # โหลดของเก่ามาก่อน แล้วรวมกับของใหม่
    current = load_registry()
    current.update(processed_files)
    
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(current), f, ensure_ascii=False, indent=4)

# --- 2. Lifespan Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_INDEX, STORAGE_CONTEXT
    print("🚀 Server Starting... Connecting to Databases...")
    
    try:
        print("🔌 Using Gemini Embedding (text-embedding-004)...")
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/text-embedding-004",
            api_key=GEMINI_API_KEY
        )

        print("🧠 Using Gemini 2.5 Flash...")
        Settings.llm = GoogleGenAI(
            model_name="models/gemini-2.5-flash", 
            api_key=GEMINI_API_KEY
        )

        # Force Check Qdrant
        print(f"🛠️ Checking Qdrant at {QDRANT_URL}...")
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )

        graph_store = Neo4jGraphStore(username=NEO4J_USER, password=NEO4J_PASS, url=NEO4J_URL)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        
        STORAGE_CONTEXT = StorageContext.from_defaults(graph_store=graph_store, vector_store=vector_store)
        print("✅ Database Connected!")

        # Load Memory
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                print("🔄 Loading existing index...")
                GLOBAL_INDEX = load_index_from_storage(STORAGE_CONTEXT, persist_dir=PERSIST_DIR)
                print("🎉 Success! Memory Loaded.")
            except Exception:
                print("⚠️ Load failed. Will rebuild on next ingest.")
                GLOBAL_INDEX = None
        else:
            print("ℹ️ No existing memory found.")
            GLOBAL_INDEX = None
            
    except Exception as e:
        print(f"❌ Startup Error: {e}")

    yield
    print("🛑 Server Shutting down...")

app = FastAPI(title="Gemini 3 Research Agent API", openapi_url="/v1/openapi.json", lifespan=lifespan)

# --- 3. Helper Functions ---
def retrieve_context(query_text: str):
    if GLOBAL_INDEX is None: return None
    try:
        retriever = GLOBAL_INDEX.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query_text)
        return "\n\n".join([n.get_content() for n in nodes]) if nodes else None
    except Exception: return None

def ask_gemini_thinking(query: str, context: Optional[str] = None):
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 
    )

    if context:
        print(f"💡 Using RAG Mode")
        prompt = f"Context:\n{context}\n\nUser Question: {query}\nAnalyze deeply."
    else:
        print("🗣️ Using General Chat Mode")
        prompt = f"User Question: {query}\nAnswer using general knowledge."

    try:
        response = client.models.generate_content(model="gemini-3-pro-preview", contents=prompt, config=config)
        thought = ""
        answer = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'thought') and part.thought: thought += part.text
            else: answer += part.text
        
        full_resp = ""
        if thought: full_resp += f"> **🧠 Thinking Process:**\n> {thought.replace(chr(10), chr(10)+'> ')}\n\n---\n\n"
        full_resp += answer
        return full_resp
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. API Endpoints ---
class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "gemini-3-researcher"
class IngestRequest(BaseModel):
    folder_path: str = "./data"

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gemini-3-researcher", "object": "model", "created": int(time.time()), "owned_by": "benzondata"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_query = request.messages[-1].content
    context_text = retrieve_context(user_query)
    reply = ask_gemini_thinking(user_query, context_text)
    return {
        "id": "chatcmpl-" + str(int(time.time())),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/ingest")
async def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    def process_ingestion(path):
        global GLOBAL_INDEX
        print(f"📂 Check Ingestion from: {path}...")
        try:
            if not os.path.exists(path): return
            
            # 1. อ่านไฟล์ทั้งหมดในโฟลเดอร์
            documents = SimpleDirectoryReader(path).load_data()
            if not documents: return

            # 2. โหลดสมุดเช็คชื่อ
            processed_files = load_registry()
            
            # 3. คัดกรอง: เอาเฉพาะไฟล์ที่ไม่เคยทำ
            new_docs = []
            new_filenames = set()
            
            for doc in documents:
                # ดึงชื่อไฟล์จาก Metadata
                file_name = os.path.basename(doc.metadata.get('file_path', 'unknown'))
                
                if file_name not in processed_files:
                    new_docs.append(doc)
                    new_filenames.add(file_name)
                # ถ้ามีแล้ว ข้ามไปเลย (ไม่ปริ้นให้รก Log)

            # 4. ถ้าไม่มีอะไรใหม่ -> จบงาน
            if not new_docs:
                print("✅ Everything is up-to-date. No new files to ingest.")
                return

            print(f"📄 Found {len(new_docs)} NEW files. Processing...")

            # 5. เริ่ม Ingest (เฉพาะไฟล์ใหม่)
            if GLOBAL_INDEX is None:
                print("🆕 Creating New Index...")
                GLOBAL_INDEX = VectorStoreIndex.from_documents(new_docs, storage_context=STORAGE_CONTEXT)
                KnowledgeGraphIndex.from_documents(new_docs, storage_context=STORAGE_CONTEXT, max_triplets_per_chunk=2, include_embeddings=True)
            else:
                print("➕ Inserting into Existing Index...")
                for doc in new_docs:
                    GLOBAL_INDEX.insert(doc)
                    # Graph Insert แบบทีละตัว
                    KnowledgeGraphIndex.from_documents([doc], storage_context=STORAGE_CONTEXT, max_triplets_per_chunk=2, include_embeddings=True)

            # 6. บันทึกสถานะ
            if not os.path.exists(PERSIST_DIR): os.makedirs(PERSIST_DIR)
            GLOBAL_INDEX.storage_context.persist(persist_dir=PERSIST_DIR)
            save_registry(new_filenames) # จดชื่อลงสมุด
            
            print(f"✅ Ingestion Complete! Added {len(new_docs)} files.")
            
        except Exception as e:
            print(f"❌ Ingestion Failed: {e}")

    background_tasks.add_task(process_ingestion, request.folder_path)
    return {"status": "Ingestion checked", "folder": request.folder_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)