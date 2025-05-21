import cv2
import os
import pickle
from insightface.app import FaceAnalysis

# Tạo thư mục lưu embeddings nếu chưa có
os.makedirs("registered_faces", exist_ok=True)

# Nhập tên người dùng
name = input("Nhập tên người cần đăng ký (viết liền không dấu): ").strip()
save_path = f"registered_faces/{name}.pkl"

if os.path.exists(save_path):
    confirm = input("Tên đã tồn tại. Ghi đè? (y/n): ")
    if confirm.lower() != 'y':
        exit()

# Load ArcFace model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # ctx_id = 0 dùng GPU, -1 dùng CPU

# Mở camera
cap = cv2.VideoCapture(0)
print("Nhấn SPACE để chụp khuôn mặt...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Đăng ký khuôn mặt", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Space
        faces = face_app.get(frame)
        if not faces:
            print("Không phát hiện khuôn mặt. Thử lại.")
            continue

        # Dùng khuôn mặt đầu tiên
        embedding = faces[0].embedding

        # Lưu embedding vào file
        with open(save_path, 'wb') as f:
            pickle.dump(embedding, f)

        print(f"✅ Đã đăng ký khuôn mặt cho: {name}")
        break

    elif key == 27:  # ESC
        print("❌ Hủy đăng ký.")
        break

cap.release()
cv2.destroyAllWindows()
