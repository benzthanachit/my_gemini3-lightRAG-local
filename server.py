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

from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, Settings, Document, load_index_from_storage
# 👇 [จุดแก้ 1] Import Gemini Embedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.gemini import Gemini
import qdrant_client

# --- 1. Configuration & Setup ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
PERSIST_DIR = "/app/storage_metadata"

GLOBAL_INDEX = None
STORAGE_CONTEXT = None

# --- 2. Lifespan Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_INDEX, STORAGE_CONTEXT
    print("🚀 Server Starting... Connecting to Databases...")
    try:
        # 👇 [จุดแก้ 2] เปลี่ยนมาใช้ Gemini Embedding (แก้ปัญหา Qdrant ว่างเปล่า/ค้าง)
        print("🔌 Using Gemini Embedding (models/text-embedding-004)...")
        Settings.embed_model = GeminiEmbedding(
            model_name="models/text-embedding-004",
            api_key=GEMINI_API_KEY
        )

        # 👇 [จุดแก้ 3] กลับไปใช้ 2.5 Flash
        Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=GEMINI_API_KEY)

        graph_store = Neo4jGraphStore(username=NEO4J_USER, password=NEO4J_PASS, url=NEO4J_URL)
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        # 👇 [จุดแก้ 4] เปลี่ยนชื่อ Collection ใหม่ เพื่อความชัวร์
        vector_store = QdrantVectorStore(client=client, collection_name="research_memory_gemini")
        
        STORAGE_CONTEXT = StorageContext.from_defaults(graph_store=graph_store, vector_store=vector_store)
        print("✅ Database Connected!")

        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                GLOBAL_INDEX = load_index_from_storage(STORAGE_CONTEXT, persist_dir=PERSIST_DIR)
                print("🎉 Success! Memory Loaded from Disk.")
            except Exception:
                GLOBAL_INDEX = None
        else:
            print("ℹ️ No existing memory found (General Chat Mode enabled).")
            GLOBAL_INDEX = None
            
    except Exception as e:
        print(f"❌ Startup Error: {e}")

    yield
    print("🛑 Server Shutting down...")

app = FastAPI(title="Gemini 3 Research Agent API", openapi_url="/v1/openapi.json", lifespan=lifespan)

# --- 3. Helper Functions ---
def retrieve_context(query_text: str):
    """ดึงข้อมูล ถ้าไม่มี Index หรือหาไม่เจอ ให้คืนค่า None"""
    if GLOBAL_INDEX is None:
        return None
    try:
        retriever = GLOBAL_INDEX.as_retriever(similarity_top_k=3, vector_store_query_mode="default")
        nodes = retriever.retrieve(query_text)
        if not nodes:
            return None
        return "\n\n".join([n.get_content() for n in nodes])
    except Exception:
        return None

def ask_gemini_thinking(query: str, context: Optional[str] = None):
    """ฟังก์ชันนี้ฉลาดขึ้น: ถ้ามี Context ก็ใช้ ถ้าไม่มีก็คุยปกติ"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 
    )

    # Logic ปรับเปลี่ยน Prompt ตามสถานการณ์
    if context:
        print("💡 Using RAG Mode (Memory Found)")
        prompt = f"""
        You are an advanced AI Researcher Assistant.
        Use the following Context from my local database to answer the user's question.
        
        --- Context ---
        {context}
        ---------------
        
        User Question: {query}
        
        Analyze the context deeply using your thinking process.
        If the context is relevant, use it.
        """
    else:
        print("🗣️ Using General Chat Mode (No Memory)")
        prompt = f"""
        You are a helpful and intelligent AI Assistant (Gemini 3).
        User Question: {query}
        
        Answer the user's question using your general knowledge and thinking process.
        """

    try:
        # ใช้ Gemini 3 Pro คิด (ทั้งแบบมีของและไม่มีของ)
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

    # 1. ลองดึงข้อมูลดูก่อน
    context_text = retrieve_context(user_query)
    
    # 2. ส่งให้ Gemini คิด (ไม่ว่าจะเจอ context หรือไม่ ก็ส่งไปหมด)
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
            
            # ใช้ Gemini Embedding สร้างทั้ง Graph และ Vector
            print(f"📄 Building Graph & Vector for {len(documents)} docs (Gemini)...")
            GLOBAL_INDEX = KnowledgeGraphIndex.from_documents(
                documents, storage_context=STORAGE_CONTEXT, max_triplets_per_chunk=2, include_embeddings=True
            )
            if not os.path.exists(PERSIST_DIR): os.makedirs(PERSIST_DIR)
            GLOBAL_INDEX.storage_context.persist(persist_dir=PERSIST_DIR)
            print("✅ Ingestion Complete!")
        except Exception as e:
            print(f"❌ Ingestion Failed: {e}")

    background_tasks.add_task(process_ingestion, request.folder_path)
    return {"status": "Ingestion started", "folder": request.folder_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)