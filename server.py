import os
import sys
import time
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from google import genai
from google.genai import types

# LlamaIndex Imports
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, VectorStoreIndex, StorageContext, Settings, Document, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

# ใช้ Google GenAI ตัวใหม่ (รองรับ Gemini 2.5)
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

# ตั้งชื่อ Collection ให้ชัดเจน
COLLECTION_NAME = "research_memory_final"

GLOBAL_INDEX = None
STORAGE_CONTEXT = None

# --- 2. Lifespan Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_INDEX, STORAGE_CONTEXT
    print("🚀 Server Starting... Connecting to Databases...")
    
    try:
        # 1. Setup Models (ใช้ SDK ตัวใหม่)
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

        # 2. Force Check/Create Qdrant Collection (Logic จาก Debug Script)
        print(f"🛠️ Force checking Qdrant at {QDRANT_URL}...")
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        try:
            if not client.collection_exists(collection_name=COLLECTION_NAME):
                print(f"   ⚠️ Collection '{COLLECTION_NAME}' not found. Creating...")
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768, # text-embedding-004 ขนาด 768
                        distance=models.Distance.COSINE
                    )
                )
                print("   ✅ Collection Created Successfully!")
            else:
                print(f"   ✅ Collection '{COLLECTION_NAME}' already exists.")
        except Exception as e:
            print(f"   ❌ Qdrant Init Warning: {e}")

        # 3. Connect LlamaIndex Components
        graph_store = Neo4jGraphStore(username=NEO4J_USER, password=NEO4J_PASS, url=NEO4J_URL)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        
        STORAGE_CONTEXT = StorageContext.from_defaults(graph_store=graph_store, vector_store=vector_store)
        print("✅ Database Connected!")

        # 4. Load Memory (พยายามโหลด Vector Index เป็นหลัก)
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                print("🔄 Loading existing index from storage...")
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
    if GLOBAL_INDEX is None:
        return None
    try:
        # ใช้ Vector Search (Qdrant)
        retriever = GLOBAL_INDEX.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query_text)
        if not nodes:
            return None
        return "\n\n".join([n.get_content() for n in nodes])
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return None

def ask_gemini_thinking(query: str, context: Optional[str] = None):
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 
    )

    if context:
        print(f"💡 Using RAG Mode ({len(context)} chars)")
        prompt = f"""
        You are an advanced AI Researcher.
        Context from database:
        {context}
        
        User Question: {query}
        Analyze the context deeply using your thinking process.
        """
    else:
        print("🗣️ Using General Chat Mode")
        prompt = f"""
        You are Gemini 3. User Question: {query}
        Answer using general knowledge.
        """

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=config
        )
        thought_text = ""
        final_answer = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'thought') and part.thought:
                thought_text += part.text
            else:
                final_answer += part.text
        
        full_response = ""
        if thought_text:
            full_response += f"> **🧠 Thinking Process:**\n> {thought_text.replace(chr(10), chr(10)+'> ')}\n\n---\n\n"
        full_response += final_answer
        return full_response
    except Exception as e:
        return f"Error from Gemini: {str(e)}"

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
    print(f"📩 Received: {user_query}")
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
        print(f"📂 Starting Ingestion from: {path}...")
        try:
            if not os.path.exists(path): return
            documents = SimpleDirectoryReader(path).load_data()
            if not documents: return
            
            print(f"📄 Processing {len(documents)} docs...")

            # 1. [สำคัญมาก] สร้าง Vector Store Index ยัดลง Qdrant โดยตรง
            # (แยกกับ Graph เพื่อความชัวร์ว่าข้อมูลเข้า)
            print("embedding into Qdrant (VectorStoreIndex)...")
            GLOBAL_INDEX = VectorStoreIndex.from_documents(
                documents, storage_context=STORAGE_CONTEXT
            )
            
            # 2. สร้าง Graph Index ลง Neo4j (ทำเสริม)
            print("building Knowledge Graph (KnowledgeGraphIndex)...")
            KnowledgeGraphIndex.from_documents(
                documents, storage_context=STORAGE_CONTEXT, max_triplets_per_chunk=2, include_embeddings=True
            )

            # Save Metadata
            if not os.path.exists(PERSIST_DIR): os.makedirs(PERSIST_DIR)
            GLOBAL_INDEX.storage_context.persist(persist_dir=PERSIST_DIR)
            
            print("✅ Ingestion Complete! Qdrant & Neo4j updated.")
            
        except Exception as e:
            print(f"❌ Ingestion Failed: {e}")

    background_tasks.add_task(process_ingestion, request.folder_path)
    return {"status": "Ingestion started", "folder": request.folder_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)