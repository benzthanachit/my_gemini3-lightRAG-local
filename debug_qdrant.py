import os
import sys
import qdrant_client
from qdrant_client.http import models
from llama_index.embeddings.ollama import OllamaEmbedding

# Setup
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
COLLECTION_NAME = "research_memory"

print("--- 🛠️ START DEBUGGING QDRANT ---")

# 1. ทดสอบเชื่อมต่อ Qdrant
try:
    print(f"1️⃣ Connecting to Qdrant at: {QDRANT_URL}")
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    print(f"   ✅ Connected! Current Collections: {names}")
except Exception as e:
    print(f"   ❌ Failed to connect to Qdrant: {e}")
    sys.exit(1)

# 2. ทดสอบ Ollama (สำคัญมาก! ถ้าอันนี้พัง Qdrant จะไม่มีข้อมูล)
try:
    print(f"\n2️⃣ Testing Ollama Embedding at: {OLLAMA_URL}")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_URL)
    # ลองแปลงคำว่า "test" เป็นตัวเลข
    vec = embed_model.get_text_embedding("test")
    vec_dim = len(vec)
    print(f"   ✅ Ollama OK! Generated Vector Dimension: {vec_dim}")
except Exception as e:
    print(f"   ❌ Ollama Failed: {e}")
    print("   👉 โปรดเช็กว่า Ollama เปิดอยู่ไหม และตั้ง OLLAMA_HOST=0.0.0.0 หรือยัง")
    sys.exit(1)

# 3. บังคับสร้าง Collection (Manual Create)
try:
    print(f"\n3️⃣ Force Creating Collection: '{COLLECTION_NAME}'")
    
    # ถ้ามีอยู่แล้ว ลบทิ้งสร้างใหม่ (เพื่อให้แน่ใจว่าสะอาด)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vec_dim, # ต้องตรงกับ Ollama (ปกติ 768)
            distance=models.Distance.COSINE
        )
    )
    print("   ✅ Collection Created Successfully!")
    
    # เช็กอีกที
    collections = client.get_collections().collections
    print(f"   🧐 Double Check: {[c.name for c in collections]}")

except Exception as e:
    print(f"   ❌ Failed to create collection: {e}")

print("\n--- ✅ DEBUG FINISHED ---")