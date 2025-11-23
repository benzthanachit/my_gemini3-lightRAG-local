# ใช้ Python Image
FROM python:3.11-slim

# ตั้ง Folder ทำงาน
WORKDIR /app

# ลง Library ที่จำเป็น
# (แนะนำให้สร้าง requirements.txt ก่อน แล้วค่อย COPY)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดทั้งหมดลงไป
COPY . .

# เปิด Port 8000
EXPOSE 8000

# รันคำสั่ง
CMD ["python", "server.py"]