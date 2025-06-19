import onnxruntime
import cv2
import numpy as np
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
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")  # Thêm để tiện truy vấn tên


class UnknownCapture(Base):
    __tablename__ = "unknown_captures"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
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


def get_embedding_from_image(image_bytes: bytes) -> List[float]:
    """
    Trích xuất embedding từ ảnh bằng model ONNX, đảm bảo nhất quán với Jetson.
    """
    try:
        # 1. Tải model ONNX
        # (Bạn có thể tải model một lần bên ngoài hàm để tối ưu hiệu năng)
        ort_session = onnxruntime.InferenceSession("exp/recognize/model-r34.onnx")
        input_name = ort_session.get_inputs()[0].name

        # 2. Đọc và tiền xử lý ảnh (giống hệt jetson_app.py)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # TODO: Cần một bước phát hiện khuôn mặt ở đây để crop lại
        # Tạm thời, ta giả định ảnh tải lên đã là ảnh khuôn mặt
        face_crop = img

        img_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, axis=0)

        # 3. Chạy inference
        ort_inputs = {input_name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)

        embedding = ort_outs[0].flatten()
        normalized_embedding = embedding / np.linalg.norm(embedding)

        return normalized_embedding.tolist()

    except Exception as e:
        print(f"Lỗi khi trích xuất embedding bằng ONNX: {e}")
        return None

# API MỚI: Xử lý việc thêm người dùng thủ công
@app.post("/api/users/manual-add")
async def manual_add_user(name: str = Form(...), image: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await image.read()

    embedding = get_embedding_from_image(image_bytes)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Không thể tìm thấy khuôn mặt trong ảnh hoặc ảnh không hợp lệ.")

    embedding_json = json.dumps(embedding)

    # Kiểm tra user đã tồn tại chưa và thêm vào CSDL
    existing_user = db.query(User).filter(User.name == name).first()
    if existing_user:
        new_embedding = FaceEmbedding(user_id=existing_user.id, embedding_json=embedding_json)
        db.add(new_embedding)
    else:
        new_user = User(name=name)
        db.add(new_user)
        db.flush()
        new_embedding = FaceEmbedding(user_id=new_user.id, embedding_json=embedding_json)
        db.add(new_embedding)

    db.commit()
    return RedirectResponse(url="/", status_code=303)


# --- Giao diện Web đơn giản (thêm bảng logs) ---
@app.get("/", response_class=HTMLResponse)
def main_page(db: Session = Depends(get_db)):
    JETSON_IP = "10.42.0.215"  # Giữ nguyên IP của Jetson

    # Lấy 10 log điểm danh gần nhất
    logs_html = ""
    logs = db.query(AttendanceLog).options(sqlalchemy.orm.joinedload(AttendanceLog.user)).order_by(AttendanceLog.timestamp.desc()).limit(10).all()
    if not logs:
        logs_html = "<li>No check-ins recorded yet.</li>"
    else:
        for log in logs:
            logs_html += f"<li>{log.user.name} checked in at {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>"

    return f"""
    <html>
        <head>
            <title>Face Recognition Dashboard</title>
            <style>
                body {{ font-family: sans-serif; display: flex; gap: 20px; padding: 10px; background-color: #f4f4f9; }}
                .column {{ padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
                .stream-container {{ flex: 2; }}
                .controls-container {{ flex: 1; }}
                .logs-container {{ flex: 1; }}
                h1, h2 {{ color: #333; border-bottom: 2px solid #5c67f2; padding-bottom: 5px; }}
                img.stream {{ border: 2px solid #333; width: 100%; max-width: 640px; background-color: #000;}}
                form div {{ margin-bottom: 15px; }}
                label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
                input[type="text"], input[type="file"] {{ width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }}
                button {{ cursor: pointer; padding: 10px 15px; background-color: #5c67f2; color: white; border: none; border-radius: 4px; font-size: 16px; }}
                button:hover {{ background-color: #4a54c2; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ background-color: #eef; padding: 8px; border-radius: 4px; margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            <div class="column stream-container">
                <h1>Live Stream from Jetson</h1>
                <img class="stream" src="http://{JETSON_IP}:8001/video_feed" alt="Live stream not available. Please check if the Jetson app is running.">
            </div>

            <div class="column controls-container">
                <h2>Add/Update User Manually</h2>
                <form action="/api/users/manual-add" method="post" enctype="multipart/form-data">
                    <div>
                        <label for="name">User Name:</label>
                        <input type="text" id="name" name="name" placeholder="e.g., Nguyen Van A" required>
                    </div>
                    <div>
                        <label for="image">Face Image:</label>
                        <input type="file" id="image" name="image" accept="image/*" required>
                    </div>
                    <div>
                        <button type="submit">Add / Update</button>
                    </div>
                </form>
                 <hr>
                <h2>All Users</h2>
                <div id="user-list"></div>
            </div>

            <div class="column logs-container">
                <h2>Recent Check-ins</h2>
                <ul id="check-in-logs">{logs_html}</ul>
            </div>

            <script>
                // Hàm để tải và hiển thị danh sách người dùng
                async function refreshUserList() {{
                    const userListDiv = document.getElementById('user-list');
                    const response = await fetch('/api/users/list');
                    const users = await response.json();

                    let userHtml = '<ul>';
                    if (users.length === 0) {{
                        userHtml += '<li>No users in the database.</li>';
                    }} else {{
                        users.forEach(user => {{
                            userHtml += `<li>${{user.name}} (ID: ${{user.id}})</li>`;
                        }});
                    }}
                    userHtml += '</ul>';
                    userListDiv.innerHTML = userHtml;
                }}

                // Tự động làm mới sau mỗi 10 giây
                setInterval(refreshUserList, 10000);

                // Chạy lần đầu khi tải trang
                document.addEventListener('DOMContentLoaded', refreshUserList);
            </script>
        </body>
    </html>
    """

class UserInfo(BaseModel):
    id: int
    name: str
    class Config:
        orm_mode = True

@app.get("/api/users/list", response_model=List[UserInfo])
def get_user_list(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.name).all()
    return users


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