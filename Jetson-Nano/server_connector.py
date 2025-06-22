import requests
import numpy as np
import cv2
import json
import config

# Cache để lưu dữ liệu khuôn mặt đã biết từ server
known_faces_cache = []

def sync_with_server():
    """Lấy danh sách khuôn mặt đã biết từ server và cập nhật cache."""
    global known_faces_cache
    try:
        response = requests.get(f"{config.SERVER_URL}/api/faces")
        if response.status_code == 200:
            data = response.json()
            for face in data:
                face['embedding'] = np.array(face['embedding'], dtype=np.float32)
            known_faces_cache = data
            print(f"Đồng bộ thành công. Có {len(known_faces_cache)} người dùng trong cache.")
        else:
            print(f"Lỗi server khi đồng bộ: {response.status_code}")
    except Exception as e:
        print(f"Lỗi kết nối khi đồng bộ: {e}")

def check_in(user_id):
    """Gửi yêu cầu điểm danh cho một user ID cụ thể."""
    try:
        requests.post(f"{config.SERVER_URL}/api/check-in", json={"user_id": user_id}, timeout=3)
        print(f"--> Đã gửi yêu cầu điểm danh cho User ID: {user_id}")
    except Exception as e:
        print(f"Lỗi khi điểm danh: {e}")

def send_unknown_capture(face_image, embedding):
    """Gửi hình ảnh và embedding của khuôn mặt lạ lên server."""
    try:
        _, img_encoded = cv2.imencode('.jpg', face_image)
        files = {'image': ('unknown.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'embedding_json': json.dumps(embedding.tolist())}
        requests.post(f"{config.SERVER_URL}/api/unknown-captures", files=files, data=data, timeout=3)
        print("--> Đã gửi khuôn mặt lạ lên server.")
    except Exception as e:
        print(f"Lỗi gửi khuôn mặt lạ: {e}")