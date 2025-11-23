import os
import sys
import time
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# FastAPI Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks

# Google GenAI
from google import genai
from google.genai import types

# LlamaIndex & DBs
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, Settings, Document, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# --- 1. Configuration & Setup ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

app = FastAPI(
    title="Gemini 3 Research Agent API",
    openapi_url="/v1/openapi.json" 
)

# Global Variable เพื่อเก็บ Index ไว้ใน Memory ของ Server
GLOBAL_INDEX = None
STORAGE_CONTEXT = None

# --- 2. Database Initialization (Run on Startup) ---
@app.on_event("startup")
def startup_event():
    global GLOBAL_INDEX, STORAGE_CONTEXT
    print("🚀 Server Starting... Connecting to Databases...")
    
    try:
        # Setup Ollama Embedding
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        Settings.llm = None 

        # Connect Neo4j & Qdrant
        graph_store = Neo4jGraphStore(username=NEO4J_USER, password=NEO4J_PASS, url=NEO4J_URL)
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(client=client, collection_name="research_memory")
        
        STORAGE_CONTEXT = StorageContext.from_defaults(graph_store=graph_store, vector_store=vector_store)
        print("✅ Database Connected!")

        # พยายามโหลด Index เก่า (ถ้ามี) - แต่ในท่า LightRAG ปกติเรามักสร้างใหม่ หรือ Load จาก Vector Store
        # เพื่อความเสถียร เราจะ Initialize เป็น None ไว้ก่อน รอ user สั่ง Ingest หรือ Load
        # ถ้าคุณมี data อยู่แล้วใน DB ก็สามารถใช้ท่า load_index_from_storage ได้ในอนาคต
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")

# --- 3. Helper Functions ---

def retrieve_context(query_text: str):
    """ฟังก์ชันดึงข้อมูลจาก Global Index"""
    if GLOBAL_INDEX is None:
        return None
    
    try:
        retriever = GLOBAL_INDEX.as_retriever(
            similarity_top_k=3, 
            vector_store_query_mode="default"
        )
        nodes = retriever.retrieve(query_text)
        if not nodes:
            return None
        return "\n\n".join([n.get_content() for n in nodes])
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return None

def ask_gemini_thinking(query: str, context: str):
    """ส่งให้ Gemini คิดและคืนค่าเป็น String"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 
    )

    prompt = f"""
    You are an advanced AI Researcher.
    
    Context from Memory:
    {context}

    User Question: {query}
    
    Analyze the context deeply using your thinking process before answering.
    Format your response in Markdown.
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
        
        # จัด Format ให้สวยงามสำหรับ WebUI
        full_response = ""
        if thought_text:
            full_response += f"> **🧠 Thinking Process:**\n> {thought_text.replace(chr(10), chr(10)+'> ')}\n\n---\n\n"
        
        full_response += final_answer
        return full_response

    except Exception as e:
        return f"Error from Gemini: {str(e)}"

# --- 4. API Models (OpenAI Compatible) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "gemini-3-researcher"
    stream: Optional[bool] = False

class IngestRequest(BaseModel):
    folder_path: str = "./data"

# --- 5. API Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Endpoint หลักสำหรับ Chat (รองรับ Open WebUI)"""
    user_query = request.messages[-1].content
    print(f"📩 Received: {user_query}")

    # 1. Retrieve
    context_text = retrieve_context(user_query)
    
    if not context_text:
        # กรณีไม่เจอข้อมูล ให้ตอบแบบปกติ หรือแจ้งเตือน
        if GLOBAL_INDEX is None:
             reply = "⚠️ ระบบยังไม่มี Memory กรุณายิง API /ingest เพื่ออ่านไฟล์ PDF ก่อนครับ"
        else:
             reply = "❌ ไม่พบข้อมูลใน Memory ที่เกี่ยวข้องกับคำถามนี้ครับ"
    else:
        # 2. Gemini Thinking
        reply = ask_gemini_thinking(user_query, context_text)

    # 3. Format Response (OpenAI Style)
    return {
        "id": "chatcmpl-" + str(int(time.time())),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": reply
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/ingest")
async def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """API สำหรับสั่งให้อ่านไฟล์ PDF ใหม่"""
    
    def process_ingestion(path):
        global GLOBAL_INDEX
        print(f"📂 Reading files from {path}...")
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print("Created data folder.")
                return

            documents = SimpleDirectoryReader(path).load_data()
            if not documents:
                print("No documents found.")
                return

            # Create Index
            GLOBAL_INDEX = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=STORAGE_CONTEXT,
                max_triplets_per_chunk=2,
                include_embeddings=True
            )
            print("✅ Ingestion Complete! Index updated.")
        except Exception as e:
            print(f"Ingestion Failed: {e}")

    # รันแบบ Background (ไม่ต้องรอให้เสร็จถึงจะตอบกลับ)
    background_tasks.add_task(process_ingestion, request.folder_path)
    return {"status": "Ingestion started in background", "folder": request.folder_path}

# --- 6. Main Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)