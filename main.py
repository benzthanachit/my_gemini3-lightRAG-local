import os
import logging
from dotenv import load_dotenv

# Google GenAI SDK (สำหรับ Gemini 3 Thinking Mode)
from google import genai
from google.genai import types

# LlamaIndex Components (สำหรับ Memory Layer)
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex, StorageContext, Settings, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password123") 
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

if not GEMINI_API_KEY:
    raise ValueError("❌ Error: ไม่พบ GEMINI_API_KEY ในไฟล์ .env")

print(f"✅ Config Loaded: Neo4j user={NEO4J_USER} at {NEO4J_URL}")

# --- 1. Setup Local Brain (Ollama for Embedding) ---
# ใช้ Ollama รันบนเครื่อง เพื่อประหยัดค่า Embedding และเก็บ Data ไว้กับตัว
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = None # เราไม่ใช้ LlamaIndex เรียก LLM ตรงๆ แต่จะเรียกผ่าน SDK Gemini 3 แทน

# --- 2. Connect to Agentic Memory (Graph + Vector) ---
# เชื่อมต่อ Graph DB (Neo4j)
graph_store = Neo4jGraphStore(
    username=NEO4J_USER,
    password=NEO4J_PASS,
    url=NEO4J_URL,
)

# เชื่อมต่อ Vector DB (Qdrant)
client = qdrant_client.QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name="research_memory")

storage_context = StorageContext.from_defaults(
    graph_store=graph_store,
    vector_store=vector_store
)

# --- Function: จำข้อมูล (Ingestion) ---
def ingest_data(text_content, doc_id):
    print(f"🧠 Generating Memory for: {doc_id}...")
    documents = [Document(text=text_content, id_=doc_id)]
    
    # สร้าง Graph Index (ความสัมพันธ์) และ Vector Index (ค้นหาความหมาย)
    # หมายเหตุ: การสร้าง Graph อัตโนมัติโดยไม่ใช้ LLM ช่วยสกัด Entity อาจจะไม่แม่นยำ 
    # ในเคสจริงเรามักใช้ Ollama (llama3) มาช่วยสกัด Entity ตรงนี้ได้
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True # Hybrid Search
    )
    print("✅ Memory Stored locally!")
    return index

# --- Function: เรียก Gemini 3 มาคิด (The Thinking Process) ---
def ask_gemini_thinking(query):
    # 1. Recall: ดึงข้อมูลจาก Local Memory ก่อน
    # (ในตัวอย่างนี้ข้ามขั้นตอน Retrieve ซับซ้อน เพื่อโชว์ส่วน Thinking)
    # สมมติเราดึง Context จาก Graph/Vector มาได้แล้ว:
    retrieved_context = "ข้อมูลงานวิจัยที่ดึงมาจาก Neo4j/Qdrant..." 

    print("🤔 Gemini 3 is thinking...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Config สำหรับ Thinking Mode
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        thinking_level="HIGH", # Maximum reasoning
        temperature=1.0 
    )

    prompt = f"""
    Context from my local database:
    {retrieved_context}

    User Question: {query}
    
    Please analyze the context deeply using your thinking process. 
    Identify connections, contradictions, or hidden patterns.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp", # หรือชื่อ model ล่าสุดที่รองรับ (เช็คชื่อใน Docs อีกที)
        contents=prompt,
        config=config
    )

    # แสดงผลความคิด (Thoughts)
    for part in response.candidates[0].content.parts:
        if part.thought:
            print(f"\n--- 💭 Thoughts Process ---\n{part.text}\n")
        else:
            print(f"\n--- 📝 Final Answer ---\n{part.text}")

# --- Main Execution ---
if __name__ == "__main__":
    # ตัวอย่าง: ป้อนข้อมูลเข้า (ทำครั้งเดียว หรือเมื่อมีข้อมูลใหม่)
    sample_text = "LightRAG เป็นเทคนิคใหม่ที่ใช้ Graph Database ร่วมกับ Vector. Gemini 3 มี Thinking Mode ที่ดีมาก."
    ingest_data(sample_text, "doc_001")

    # ตัวอย่าง: ถามคำถาม
    ask_gemini_thinking("LightRAG กับ Gemini 3 ทำงานร่วมกันได้อย่างไร?")