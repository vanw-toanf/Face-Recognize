import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine


def load_registered_faces(folder="registered_faces"):
    """
    Load tất cả vector embedding của người đã đăng ký từ thư mục.
    Trả về dict: {name: embedding_vector}
    """
    embeddings = {}
    if not os.path.exists(folder):
        os.makedirs(folder)

    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            name = file[:-4]
            with open(os.path.join(folder, file), 'rb') as f:
                embeddings[name] = pickle.load(f)
    return embeddings


def recognize_face(face_img, recognizer, known_faces, threshold=0.5):
    """
    Trích xuất embedding từ ảnh mặt và so sánh với known_faces.
    Trả về tên nếu khớp, nếu không thì "Unknown".
    """
    faces = recognizer.get(face_img)
    if not faces:
        return "Unknown"

    emb = faces[0].embedding
    best_score = 1.0
    identity = "Unknown"

    for name, known_emb in known_faces.items():
        score = cosine(emb, known_emb)
        if score < best_score and score < threshold:
            best_score = score
            identity = name

    return identity
