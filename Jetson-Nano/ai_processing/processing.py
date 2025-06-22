import numpy as np
import cv2
import config

def preprocess_yolo(img):
    """Tiền xử lý ảnh cho model YOLO (letterbox, chuẩn hóa, chuyển layout)."""
    h, w, _ = img.shape
    ih, iw = config.YOLO_INPUT_SHAPE
    scale = min(ih / h, iw / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    new_img = np.full((ih, iw, 3), 114, dtype=np.uint8)
    dx = (iw - nw) // 2
    dy = (ih - nh) // 2
    new_img[dy:dy + nh, dx:dx + nw, :] = img_resized

    img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)
    return input_tensor, scale, dx, dy

def postprocess_yolo(output, original_shape):
    """Hậu xử lý output của YOLO (NMS, scale tọa độ)."""
    h, w = original_shape
    output = output.reshape(1, -1, 6)
    boxes, confidences = [], []

    for det in output[0]:
        confidence = det[4]
        if confidence >= config.CONF_THRESHOLD:
            cx, cy, bw, bh = det[:4]
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))

    if not boxes: return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONF_THRESHOLD, config.NMS_THRESHOLD)
    if len(indices) == 0: return []

    final_boxes = []
    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        scale, dx, dy = postprocess_yolo.scale_params
        x1_orig = int((x - dx) / scale)
        y1_orig = int((y - dy) / scale)
        x2_orig = int((x + bw - dx) / scale)
        y2_orig = int((y + bh - dy) / scale)

        final_boxes.append([
            max(0, x1_orig), max(0, y1_orig),
            min(w, x2_orig), min(h, y2_orig),
            confidences[i]
        ])
    return final_boxes

def preprocess_arcface(face_img):
    """Tiền xử lý ảnh khuôn mặt cho ArcFace."""
    img_resized = cv2.resize(face_img, config.RECOGNIZER_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)
    return input_tensor

def calculate_iou(box1, box2):
    """Tính toán chỉ số Intersection over Union (IoU)."""
    x1_1, y1_1, x2_1, y2_1, _ = box1
    x1_2, y1_2, x2_2, y2_2, _ = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0