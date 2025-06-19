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


def sigmoid(x):
    """
    Hàm sigmoid.
    """
    return 1 / (1 + np.exp(-x))

def decode_output(output, img_w, img_h, conf_thresh=0.5, iou_thresh=0.4):
    """
    Giải mã output TensorRT YOLO thành list box.

    Args:
        output: numpy array, shape (N, 85) hoặc (N, 6)
        img_w, img_h: kích thước ảnh gốc (để scale lại)
        conf_thresh: ngưỡng lọc confidence
        iou_thresh: ngưỡng NMS (nếu muốn)

    Returns:
        boxes: list (x1, y1, x2, y2, conf, class_id)
    """
    # Nếu output có batch dim, loại bỏ (giả sử 1 batch)
    if len(output.shape) == 3:
        output = output[0]

    boxes = []
    # Giả sử output dạng (num_boxes, 85)
    # 4 bbox + 1 obj_conf + 80 class conf
    # Nếu 6 thì 4 bbox + 1 conf + 1 class

    for pred in output:
        if pred.shape[0] == 85:
            # bbox center_x, center_y, w, h
            cx, cy, w, h = pred[0:4]
            object_conf = sigmoid(pred[4])
            class_probs = sigmoid(pred[5:])
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            conf = object_conf * class_conf

            if conf < conf_thresh:
                continue

            # Chuyển từ center_x, center_y, w, h sang x1, y1, x2, y2
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            # Giới hạn tọa độ trong ảnh
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            boxes.append([x1, y1, x2, y2, conf, class_id])

        elif pred.shape[0] == 6:
            cx, cy, w, h = pred[0:4]
            conf = sigmoid(pred[4])
            class_id = int(pred[5])

            if conf < conf_thresh:
                continue

            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            boxes.append([x1, y1, x2, y2, conf, class_id])

    # TODO: bạn có thể chạy NMS để loại box trùng (khuyến khích)

    return boxes