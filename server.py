import sqlalchemy
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, LargeBinary, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import io
import json
import base64
from starlette.requests import Request
from datetime import date, datetime, time

DATABASE_URL = "sqlite:///./attendance.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Định nghĩa Models cho CSDL (Giữ nguyên) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    embeddings = relationship("FaceEmbedding", back_populates="owner", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding_json = Column(String)  # Lưu embedding dưới dạng chuỗi JSON
    owner = relationship("User", back_populates="embeddings")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User")  # Thêm để tiện truy vấn tên


class UnknownCapture(Base):
    __tablename__ = "unknown_captures"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_data = Column(LargeBinary)  # Lưu dữ liệu ảnh
    embedding_json = Column(String)  # Lưu embedding của ảnh này


Base.metadata.create_all(bind=engine)


# --- Pydantic Schemas (Định nghĩa cấu trúc dữ liệu API) ---
class CheckInRequest(BaseModel):
    user_id: int


# <<< THAY ĐỔI Ở ĐÂY: Tạo schema mới phù hợp với client Jetson ---
class FaceCacheData(BaseModel):
    id: int  # ID của user
    name: str
    embedding: List[float]


# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()


# --- API Endpoints ---
# <<< THAY ĐỔI Ở ĐÂY: Sửa lại hàm và response_model ---
@app.get("/api/faces", response_model=List[FaceCacheData])
def get_all_faces(db: Session = Depends(get_db)):
    """
    API cho Jetson gọi để đồng bộ dữ liệu.
    Đã được "trải phẳng" để mỗi embedding là một phần tử riêng biệt.
    """
    users = db.query(User).options(sqlalchemy.orm.joinedload(User.embeddings)).all()
    result = []
    for user in users:
        for emb in user.embeddings:
            result.append(FaceCacheData(
                id=user.id,
                name=user.name,
                embedding=json.loads(emb.embedding_json)
            ))
    return result


@app.post("/api/check-in")
def record_check_in(request: CheckInRequest, db: Session = Depends(get_db)):
    """API cho Jetson gọi để ghi nhận chấm công."""
    today = date.today()
    start_of_day = datetime.combine(today, time.min)
    end_of_day = datetime.combine(today, time.max)

    # Kiểm tra xem có log nào của user này trong ngày hôm nay chưa
    existing_log = db.query(AttendanceLog).filter(
        AttendanceLog.user_id == request.user_id,
        AttendanceLog.timestamp >= start_of_day,
        AttendanceLog.timestamp <= end_of_day
    ).first()

    if existing_log:
        # Nếu đã có, không làm gì cả và báo lại
        return {"status": "already_checked_in_today", "user_id": request.user_id}
    else:
        # Nếu chưa có, tạo log mới
        db_log = AttendanceLog(user_id=request.user_id, timestamp=datetime.now())
        db.add(db_log)
        db.commit()
        return {"status": "check_in_successful", "user_id": request.user_id}


@app.post("/api/unknown-captures")
async def upload_unknown_face(embedding_json: str = Form(...), image: UploadFile = File(...),
                              db: Session = Depends(get_db)):
    """API cho Jetson gửi ảnh và embedding của người lạ lên."""
    image_bytes = await image.read()
    new_capture = UnknownCapture(image_data=image_bytes, embedding_json=embedding_json)
    db.add(new_capture)
    db.commit()
    return {"status": "capture received", "size": len(image_bytes)}


@app.post("/api/users/add-from-capture")
async def add_user_from_capture(request: Request, db: Session = Depends(get_db)):
    """API được gọi từ giao diện web để thêm người dùng mới từ một capture."""
    form_data = await request.form()
    capture_id = int(form_data.get("capture_id"))
    name = form_data.get("name")

    capture = db.query(UnknownCapture).filter(UnknownCapture.id == capture_id).first()
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")

    existing_user = db.query(User).filter(User.name == name).first()
    if existing_user:
        new_embedding = FaceEmbedding(user_id=existing_user.id, embedding_json=capture.embedding_json)
        db.add(new_embedding)
    else:
        new_user = User(name=name)
        db.add(new_user)
        db.flush()  # Để lấy user.id
        new_embedding = FaceEmbedding(user_id=new_user.id, embedding_json=capture.embedding_json)
        db.add(new_embedding)

    db.delete(capture)
    db.commit()

    # Chuyển hướng về trang chủ sau khi xử lý
    return RedirectResponse(url="/", status_code=303)


# --- Giao diện Web đơn giản (thêm bảng logs) ---
@app.get("/", response_class=HTMLResponse)
def main_page(db: Session = Depends(get_db)):
    JETSON_IP = "10.42.0.215"

    # Lấy 10 log điểm danh gần nhất
    logs_html = ""
    logs = db.query(AttendanceLog).order_by(AttendanceLog.timestamp.desc()).limit(10).all()
    for log in logs:
        logs_html += f"<li>{log.user.name} checked in at {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>"

    return f"""
    <html>
        <head>
            <title>Face Recognition Dashboard</title>
            <style>
                body {{ font-family: sans-serif; display: flex; gap: 20px; }}
                .column {{ padding: 10px; border: 1px solid #ddd; border-radius: 8px; }}
                .stream-container {{ flex: 2; }}
                .controls-container {{ flex: 1; }}
                .logs-container {{ flex: 1; }}
                h1, h2 {{ color: #333; }}
                img.stream {{ border: 2px solid #333; width: 100%; max-width: 640px; background-color: #000;}}
                .capture-item {{ border: 1px solid #ccc; margin-bottom: 10px; padding: 10px; display: flex; align-items: center; gap: 15px;}}
                .capture-item img {{ width: 112px; height: 112px; }}
                .capture-item form {{ display: flex; flex-direction: column; gap: 5px;}}
                button {{ cursor: pointer; padding: 5px 10px; }}
            </style>
        </head>
        <body>
            <div class="column stream-container">
                <h1>Live Stream from Jetson</h1>
                <img class="stream" src="http://{JETSON_IP}:8001/video_feed" alt="Live stream not available.">
            </div>
            <div class="column controls-container">
                <h2>Unknown Faces</h2>
                <div id="unknown-faces-list">Loading...</div>
            </div>
            <div class="column logs-container">
                <h2>Recent Check-ins</h2>
                <ul id="check-in-logs">{logs_html}</ul>
            </div>
            <script>
                async function refreshUnknownFaces() {{
                    const listDiv = document.getElementById('unknown-faces-list');
                    try {{
                        const response = await fetch('/api/unknown-captures/view');
                        if (!response.ok) {{ throw new Error('Network response was not ok'); }}
                        const captures = await response.json();
                        listDiv.innerHTML = '';
                        if (captures.length === 0) {{
                            listDiv.innerHTML = '<p>No unknown faces detected recently.</p>';
                        }} else {{
                            captures.forEach(capture => {{
                                const item = document.createElement('div');
                                item.className = 'capture-item';
                                item.innerHTML = `
                                    <img src="data:image/jpeg;base64,${{capture.image_base64}}" alt="Unknown Face" />
                                    <div>
                                        <p>Capture ID: ${{capture.id}} <br/> <small>${{new Date(capture.timestamp).toLocaleString()}}</small></p>
                                        <form onsubmit="this.querySelector('button').disabled=true;" action="/api/users/add-from-capture" method="post" target="_self">
                                            <input type="hidden" name="capture_id" value="${{capture.id}}" />
                                            <input type="text" name="name" placeholder="Enter name" required />
                                            <button type="submit">Add User</button>
                                        </form>
                                    </div>
                                `;
                                listDiv.appendChild(item);
                            }});
                        }}
                    }} catch (error) {{
                        listDiv.innerHTML = '<p>Error loading unknown faces. Is the server running correctly?</p>';
                        console.error('Error fetching unknown faces:', error);
                    }}
                }}

                // Tự động làm mới 
                setInterval(refreshUnknownFaces, 5000); // Làm mới mỗi 5 giây

                // Chạy lần đầu khi tải trang
                document.addEventListener('DOMContentLoaded', refreshUnknownFaces);
            </script>
        </body>
    </html>
    """


# API phụ để giao diện web lấy ảnh (chuyển sang base64)
@app.get("/api/unknown-captures/view")
def view_unknown_captures(db: Session = Depends(get_db)):
    captures = db.query(UnknownCapture).order_by(UnknownCapture.timestamp.desc()).limit(10).all()
    result = []
    for cap in captures:
        result.append({
            "id": cap.id,
            "timestamp": cap.timestamp,
            "image_base64": base64.b64encode(cap.image_data).decode('utf-8')
        })
    return result