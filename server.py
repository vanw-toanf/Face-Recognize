from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

# Thư mục lưu log
os.makedirs("received_logs", exist_ok=True)

class LogEntry(BaseModel):
    name: str
    timestamp: str

@app.post("/log")
async def receive_log(entry: LogEntry):
    log_path = "received_logs/attendance_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{entry.timestamp} - {entry.name}\n")
    print(f"✅ Nhận log: {entry.timestamp} - {entry.name}")
    return {"status": "received"}
