import os
import sys
from dotenv import load_dotenv

# --- Library สำหรับ Google Gemini 3 (Thinking Mode) ---
from google import genai
from google.genai import types

# --- Library สำหรับ Agentic Memory (LlamaIndex) ---
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# 1. โหลด Config จากไฟล์ .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# ตรวจสอบความปลอดภัย
if not GEMINI_API_KEY:
    print("❌ Error: ไม่พบ GEMINI_API_KEY ในไฟล์ .env")
    sys.exit(1)

print(f"✅ Config Loaded: Connecting to Neo4j at {NEO4J_URL}...")

# 2. Setup Local Brain (Ollama)
# ใช้ Ollama (Model: nomic-embed-text) ทำหน้าที่เปลี่ยนข้อความเป็นตัวเลข (Embedding)
# รันบน PC ของคุณเอง ฟรีและเป็นส่วนตัว
try:
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = None # ปิด LLM ของ LlamaIndex เพราะเราจะใช้ Gemini SDK ยิงตรง
except Exception as e:
    print(f"⚠️ Warning: เชื่อมต่อ Ollama ไม่ได้ (ตรวจสอบว่ารัน 'ollama serve' หรือยัง): {e}")

# 3. Connect to Databases (Agentic Memory)
try:
    # A. Graph Database (Neo4j) - เก็บความสัมพันธ์
    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASS,
        url=NEO4J_URL,
    )
    
    # B. Vector Database (Qdrant) - เก็บความหมายเพื่อค้นหา
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name="research_memory")

    # รวม 2 Database เข้าด้วยกันเป็น Storage เดียว
    storage_context = StorageContext.from_defaults(
        graph_store=graph_store, 
        vector_store=vector_store
    )
except Exception as e:
    print(f"❌ Database Connection Error: {e}")
    sys.exit(1)


# --- Function 1: จำข้อมูล (Ingestion) ---
def ingest_data(text_content, doc_id):
    """นำข้อความดิบ เข้าไปเก็บใน Graph และ Vector Database"""
    print(f"\n📥 Ingesting data ID: {doc_id}...")
    documents = [Document(text=text_content, id_=doc_id)]
    
    # สร้าง Index (ขั้นตอนนี้จะใช้เวลาสักครู่ เพื่อยิง Embedding และสร้าง Graph)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True # เปิด Hybrid Search (Vector + Graph)
    )
    print("✅ Memory Stored locally in Neo4j & Qdrant!")
    return index


# --- Function 2: ค้นหาข้อมูล (Retrieval) ---
def retrieve_data(index, query_text):
    """ค้นหาข้อมูลจาก Memory ที่เกี่ยวข้องกับคำถาม"""
    print(f"\n🔍 Searching memory for: '{query_text}'...")
    
    # สร้าง Retriever ให้ค้นหาทั้งแบบ Vector และ Keyword
    retriever = index.as_retriever(
        similarity_top_k=3, # ดึงมา 3 ชิ้นที่เกี่ยวข้องที่สุด
        vector_store_query_mode="default" 
    )
    
    nodes = retriever.retrieve(query_text)
    
    if not nodes:
        return None
        
    # รวมเนื้อหาที่เจอเป็นก้อนข้อความเดียว
    context_text = "\n\n".join([n.get_content() for n in nodes])
    print(f"📄 Found {len(nodes)} relevant context snippets.")
    return context_text


# --- Function 3: ถาม Gemini 3 (Thinking Process) ---
def ask_gemini_thinking(query, context_text):
    """ส่งข้อมูลให้ Gemini 3 คิดวิเคราะห์"""
    print("\n🤔 Gemini 3 is thinking...")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Config: เปิด Thinking Mode (ตัด thinking_level ออกเพื่อแก้ Error Pydantic)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 
    )

    # Prompt: สั่งให้ตอบจาก Context เท่านั้น
    prompt = f"""
    You are an advanced research assistant. 
    Analyze the provided context deeply to answer the user's question.
    
    --- Context from Local Database ---
    {context_text}
    -----------------------------------

    User Question: {query}
    
    Please use your thinking process to identify connections and provide a comprehensive answer.
    """

    try:
        # เรียกใช้ Gemini 3
        response = client.models.generate_content(
            model="gemini-3-pro-preview",  # ใช้ชื่อโมเดลตามที่คุณต้องการ
            contents=prompt,
            config=config
        )

        # แสดงผลลัพธ์ (แยกส่วนความคิดกับคำตอบ)
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'thought') and part.thought:
                print(f"\n--- 💭 Thoughts Process ---\n{part.text}\n")
            else:
                print(f"\n--- 📝 Final Answer ---\n{part.text}")
                
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. ข้อมูลตัวอย่าง (สมมติว่าเป็นเนื้อหาจาก Paper)
    # คุณสามารถเปลี่ยนตรงนี้เป็นการอ่านไฟล์ PDF ได้ในอนาคต
    research_text = """
    LightRAG เป็นสถาปัตยกรรม Retrieval-Augmented Generation แบบใหม่ที่ผสาน Graph Database เข้ากับ Vector Search.
    ข้อดีของ LightRAG คือช่วยให้ LLM เข้าใจบริบทเชิงโครงสร้าง (Structural Context) ได้ดีกว่า RAG ทั่วไป.
    Gemini 3 Pro มาพร้อมฟีเจอร์ Thinking Mode ที่สามารถวางแผน (Planning) และตรวจสอบเหตุผล (Reasoning) ได้ลึกซึ้ง.
    การใช้ LightRAG คู่กับ Gemini 3 ช่วยลดปัญหา Hallucination ในงานวิจัยวิทยาศาสตร์ได้ 40%.
    """
    
    # 2. นำเข้าข้อมูล (Ingest) -> เก็บลง Neo4j/Qdrant
    # หมายเหตุ: ถ้าข้อมูลเดิมมีอยู่แล้ว มันอาจจะสร้างซ้ำ ในงานจริงเราจะเช็คก่อน
    index = ingest_data(research_text, "doc_research_001")

    # 3. ตั้งคำถาม
    user_query = "LightRAG ช่วยลดปัญหาอะไร และทำไมต้องใช้คู่กับ Gemini 3?"

    # 4. ดึงข้อมูลจริง (Retrieve)
    real_context = retrieve_data(index, user_query)

    # 5. ส่งให้ Gemini คิด (Generate)
    if real_context:
        ask_gemini_thinking(user_query, real_context)
    else:
        print("❌ ไม่พบข้อมูลที่เกี่ยวข้องใน Database")