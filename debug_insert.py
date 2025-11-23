import os
import random
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Config (เอาให้ตรงกับใน Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "test_manual_insert" # สร้างชื่อใหม่เลย เอาให้ชัวร์

print("--- 💥 START NUCLEAR TEST: MANUAL INSERT ---")
print(f"🎯 Target: {QDRANT_URL}")

try:
    # 1. เชื่อมต่อ
    client = QdrantClient(url=QDRANT_URL)
    print("✅ Connected to Qdrant")

    # 2. สร้าง Collection (Vector Size 768 คือมาตรฐานของ Nomic/Gemini)
    print(f"🛠️ Recreating collection: {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768, 
            distance=models.Distance.COSINE
        )
    )
    print("✅ Collection Created")

    # 3. สร้างข้อมูลจำลอง (ไม่ต้องพึ่ง Embedding Model เดี๋ยวจะพาล error)
    # เราสุ่มตัวเลขขึ้นมา 768 ตัว เพื่อจำลองว่าเป็น Vector
    dummy_vector = [random.random() for _ in range(768)]
    
    payload_data = {
        "text": "นี่คือข้อความทดสอบจาก Benzon Lab (Manual Insert)",
        "meta": "test_data",
        "status": "active"
    }

    print("📤 Inserting 1 Point...")
    # 4. ยัดข้อมูลลงไปตรงๆ (Upsert)
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector=dummy_vector,
                payload=payload_data
            )
        ]
    )
    print(f"✅ Insert Status: {operation_info.status}")

    # 5. ตรวจสอบทันที (Count)
    count_result = client.count(collection_name=COLLECTION_NAME)
    print(f"📊 Total Points in DB: {count_result.count}")

    if count_result.count > 0:
        print("\n🎉 SUCCESS! Qdrant is WRITABLE.")
        print("ปัญหาน่าจะอยู่ที่ LlamaIndex Config ไม่ใช่ที่ Database")
    else:
        print("\n💀 FAILED! Qdrant is not saving data.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")