import json
import base64
from datetime import date, datetime, time
from typing import List

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import sqlalchemy

from . import models, schemas, services
from .database import engine, get_db

# Tạo các bảng trong CSDL nếu chưa tồn tại
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# === GIAO DIỆN WEB ===

@app.get("/", response_class=HTMLResponse)
def main_page(request: Request, db: Session = Depends(get_db)):
    logs = db.query(models.AttendanceLog).options(sqlalchemy.orm.joinedload(models.AttendanceLog.user)).order_by(models.AttendanceLog.timestamp.desc()).limit(20).all()
    users = db.query(models.User).order_by(models.User.name).all()
    return templates.TemplateResponse("index.html", {"request": request, "logs": logs, "users": users})

# === API CHO JETSON ===

@app.get("/api/faces", response_model=List[schemas.FaceCacheData])
def get_all_faces(db: Session = Depends(get_db)):
    users = db.query(models.User).options(sqlalchemy.orm.joinedload(models.User.embeddings)).all()
    result = [
        schemas.FaceCacheData(id=user.id, name=user.name, embedding=json.loads(emb.embedding_json))
        for user in users for emb in user.embeddings
    ]
    return result

@app.post("/api/check-in")
def record_check_in(request: schemas.CheckInRequest, db: Session = Depends(get_db)):
    today_start = datetime.combine(date.today(), time.min)
    existing_log = db.query(models.AttendanceLog).filter(
        models.AttendanceLog.user_id == request.user_id,
        models.AttendanceLog.timestamp >= today_start
    ).first()

    if existing_log:
        return {"status": "already_checked_in_today", "user_id": request.user_id}

    db_log = models.AttendanceLog(user_id=request.user_id, timestamp=datetime.now())
    db.add(db_log)
    db.commit()
    return {"status": "check_in_successful", "user_id": request.user_id}

# === API CHO GIAO DIỆN WEB ===

@app.post("/api/users/manual-add")
async def manual_add_user(name: str = Form(...), image: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await image.read()
    embedding = services.get_embedding_from_image(image_bytes)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Không thể trích xuất embedding từ ảnh.")

    embedding_json = json.dumps(embedding)
    existing_user = db.query(models.User).filter(models.User.name == name).first()

    if existing_user:
        new_embedding = models.FaceEmbedding(user_id=existing_user.id, embedding_json=embedding_json)
        db.add(new_embedding)
    else:
        new_user = models.User(name=name)
        db.add(new_user)
        db.flush()
        new_embedding = models.FaceEmbedding(user_id=new_user.id, embedding_json=embedding_json)
        db.add(new_embedding)

    db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.post("/api/users/delete/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user_to_delete = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

    # Nhờ có cascade="all, delete-orphan", các log và embedding liên quan sẽ tự động bị xóa.
    db.delete(user_to_delete)
    db.commit()
    return RedirectResponse(url="/", status_code=303)