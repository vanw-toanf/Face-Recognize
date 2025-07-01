from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Dùng cho API đồng bộ với Jetson
class FaceCacheData(BaseModel):
    id: int
    name: str
    embedding: List[float]

# Dùng cho API điểm danh từ Jetson
class CheckInRequest(BaseModel):
    user_id: int

# Dùng cho API hiển thị danh sách user trên web
class UserInfo(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True

# Dùng cho API hiển thị các ảnh chưa nhận dạng
class UnknownCaptureInfo(BaseModel):
    id: int
    timestamp: datetime
    image_base64: str