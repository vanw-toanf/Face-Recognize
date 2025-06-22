from sqlalchemy import (Column, Integer, String, Float, DateTime,
                        LargeBinary, ForeignKey)
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    # Khi một User bị xóa, tất cả FaceEmbedding liên quan cũng sẽ bị xóa
    embeddings = relationship("FaceEmbedding", back_populates="owner", cascade="all, delete-orphan")
    # Mối quan hệ với logs để có thể xóa khi cần
    logs = relationship("AttendanceLog", back_populates="user", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding_json = Column(String)

    owner = relationship("User", back_populates="embeddings")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logs")


class UnknownCapture(Base):
    __tablename__ = "unknown_captures"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_data = Column(LargeBinary)
    embedding_json = Column(String)