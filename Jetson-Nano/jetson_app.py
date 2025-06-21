import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import json
import requests
import threading
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import io
import uvicorn
import math
import asyncio

# --- CẤU HÌNH ---
SERVER_IP = "10.42.0.1"
SERVER_URL = "http://" + SERVER_IP + ":8000"
YOLO_ENGINE_PATH = "model/detect/best.engine"
RECOGNIZER_ENGINE_PATH = "model/recognize/model-r34.engine"
YOLO_INPUT_SHAPE = (320, 320)  # (height, width)
RECOGNIZER_INPUT_SHAPE = (112, 112)  # (height, width)
CONF_THRESHOLD = 0.5  # Ngưỡng tin cậy cho YOLO
NMS_THRESHOLD = 0.4  # Ngưỡng cho Non-Maximum Suppression
COSINE_THRESHOLD = 0.35  # Ngưỡng nhận dạng, cần tinh chỉnh sau khi test
frame_counter = 0
RECOGNITION_INTERVAL = 25 # Chỉ nhận dạng 1 lần mỗi 25 frames
face_identities = {} # Lưu trữ ID và tên của các khuôn mặt đang được theo dõi
IOU_THRESHOLD = 0.4


# --- Lớp Wrapper cho TensorRT ---
class TRT_Engine:
    def __init__(self, engine_path, max_batch_size=1):
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for binding in self.engine:
            # Tính toán size bộ nhớ dựa trên max_batch_size
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Phân bổ bộ nhớ trên Host (CPU) và Device (GPU)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def __call__(self, host_input: np.ndarray):
        # Lấy kích thước batch thực tế từ input
        batch_size = host_input.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Kích thước batch đầu vào ({batch_size}) lớn hơn max_batch_size đã cấu hình ({self.max_batch_size})")

        # Chuẩn bị input và output buffers cho kích thước batch thực tế
        host_input = np.ascontiguousarray(host_input)

        # Sao chép dữ liệu từ CPU sang GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], host_input, self.stream)

        # Đặt kích thước batch cho context nếu cần (cho các model có dynamic shape)
        # self.context.set_binding_shape(0, host_input.shape)

        # Thực thi model
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Tính toán kích thước output thực tế
        output_shape = tuple([batch_size] + list(self.outputs[0]['shape'][1:]))

        # Lấy kết quả từ GPU về CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Reshape và trả về đúng kích thước output
        return self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)


# --- Các hàm tiền xử lý và hậu xử lý ---

def preprocess_yolo(img, input_size=(640, 640)):
    """
    Tiền xử lý ảnh cho model YOLOv5.
    - Resize ảnh với padding (letterbox) để giữ nguyên tỉ lệ.
    - Chuẩn hóa giá trị pixel về [0, 1].
    - Chuyển layout từ HWC sang CHW.
    """
    h, w, _ = img.shape
    ih, iw = input_size
    scale = min(ih / h, iw / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Tạo một ảnh nền màu xám và dán ảnh đã resize vào giữa
    new_img = np.full((ih, iw, 3), 114, dtype=np.uint8)
    dx = (iw - nw) // 2
    dy = (ih - nh) // 2
    new_img[dy:dy + nh, dx:dx + nw, :] = img_resized

    # Chuẩn hóa
    img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Chuyển layout HWC to CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # Thêm chiều batch
    input_tensor = np.expand_dims(img_chw, axis=0)

    return input_tensor, scale, dx, dy


def postprocess_yolo(output, conf_thres, nms_thres, original_shape, scale, dx, dy):
    """
    Hậu xử lý output của YOLOv5 để lấy bounding boxes.
    - Áp dụng Non-Maximum Suppression.
    - Scale bounding boxes về kích thước ảnh gốc.
    """
    h, w = original_shape

    # Output của YOLOv5 TensorRT: (1, số_box, 6) [x, y, w, h, conf, class_id]
    # Hoặc (1, 6, số_box) -> cần reshape
    output = output.reshape(1, -1, 6)  # có 1 class là "face"

    boxes, confidences = [], []
    for det in output[0]:
        confidence = det[4]
        if confidence >= conf_thres:
            # Chuyển [center_x, center_y, width, height] -> [x1, y1, x2, y2]
            cx, cy, bw, bh = det[:4]
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # cv2.dnn.NMSBoxes cần [x, y, w, h]
            confidences.append(float(confidence))

    if not boxes:
        return []

    # Áp dụng NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    if len(indices) == 0:
        return []

    final_boxes = []
    for i in indices.flatten():
        x, y, bw, bh = boxes[i]

        # Scale tọa độ về ảnh gốc
        x1_orig = int((x - dx) / scale)
        y1_orig = int((y - dy) / scale)
        x2_orig = int((x + bw - dx) / scale)
        y2_orig = int((y + bh - dy) / scale)

        # Đảm bảo box nằm trong ảnh
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(w, x2_orig)
        y2_orig = min(h, y2_orig)

        final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig, confidences[i]])

    return final_boxes


def preprocess_arcface(face_img):
    """
    Tiền xử lý ảnh khuôn mặt cho ArcFace.
    """
    img_resized = cv2.resize(face_img, RECOGNIZER_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)
    return input_tensor


def calculate_iou(box1, box2):
    """
    Tính toán chỉ số Intersection over Union (IoU) giữa hai bounding box.
    Cả hai box đều có 5 phần tử [x1, y1, x2, y2, confidence].
    """
    # Giải nén 5 giá trị và bỏ qua giá trị cuối cùng (confidence) bằng dấu gạch dưới "_"
    x1_1, y1_1, x2_1, y2_1, _ = box1
    x1_2, y1_2, x2_2, y2_2, _ = box2

    # Tính toán diện tích phần giao nhau
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Tính toán diện tích của mỗi box
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Tính toán diện tích phần hợp
    union_area = box1_area + box2_area - inter_area

    # Tính IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# --- Các hàm giao tiếp Server ---
known_faces_cache = []


def sync_with_server():
    global known_faces_cache
    try:
        response = requests.get(f"{SERVER_URL}/api/faces")
        if response.status_code == 200:
            data = response.json()
            # Chuyển embedding từ list sang numpy array
            for face in data:
                face['embedding'] = np.array(face['embedding'], dtype=np.float32)
            known_faces_cache = data
            print(f"Đồng bộ thành công. Có {len(known_faces_cache)} người dùng trong cache.")
        else:
            print(f"Lỗi server khi đồng bộ: {response.status_code}")
    except Exception as e:
        print(f"Lỗi kết nối khi đồng bộ: {e}")


def check_in(user_id):
    try:
        requests.post(f"{SERVER_URL}/api/check-in", json={"user_id": user_id}, timeout=3)
        print(f"--> Đã gửi yêu cầu điểm danh cho User ID: {user_id}")
    except Exception as e:
        print(f"Lỗi khi điểm danh: {e}")


def send_unknown_capture(face_image, embedding):
    try:
        _, img_encoded = cv2.imencode('.jpg', face_image)
        files = {'image': ('unknown.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'embedding_json': json.dumps(embedding.tolist())}
        requests.post(f"{SERVER_URL}/api/unknown-captures", files=files, data=data, timeout=3)
        print("--> Đã gửi khuôn mặt lạ lên server.")
    except Exception as e:
        print(f"Lỗi gửi khuôn mặt lạ: {e}")


# --- Logic cho ứng dụng FastAPI ---
app = FastAPI()
latest_processed_frame = None
is_running = True


def run_streaming_server():
    # tạo và set event loop cho luồng mới
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


@app.get("/video_feed")
def video_feed():
    def generate():
        while is_running:
            if latest_processed_frame is not None:
                (flag, encodedImage) = cv2.imencode(".jpg", latest_processed_frame)
                if not flag: continue
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- Hàm AI chính ---
def ai_processing_loop():
    global latest_processed_frame, is_running

    print("Đang tải model...")
    yolo_engine = TRT_Engine(YOLO_ENGINE_PATH, max_batch_size=1)
    arcface_engine = TRT_Engine(RECOGNIZER_ENGINE_PATH, max_batch_size=5)  # Cho phép xử lý tối đa 5 khuôn mặt 1 lúc
    print("Tải model thành công!")

    frame_counter = 0
    tracked_faces = {}  # {track_id: {box, name, user_id, last_seen_frame}}
    next_track_id = 0

    # --- Các biến cho giao tiếp server ---
    sync_with_server()
    last_sync_time = time.time()
    last_check_in = {}
    CHECK_IN_COOLDOWN = 10

    # --- Khởi tạo Camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.");
        is_running = False;
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- Biến tính FPS ---
    fps_start_time = time.time()
    fps_frame_count = 0

    while is_running:
        if time.time() - last_sync_time > 60:
            sync_with_server()
            last_sync_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Hết frame hoặc camera bị ngắt.");
            break

        frame_counter += 1

        yolo_input, scale, dx, dy = preprocess_yolo(frame, YOLO_INPUT_SHAPE)
        yolo_output = yolo_engine(yolo_input)
        current_boxes = postprocess_yolo(yolo_output, CONF_THRESHOLD, NMS_THRESHOLD, frame.shape[:2], scale, dx, dy)
        current_boxes = sorted(current_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:5]

        # --- Logic Tracking và Nhận dạng ---
        matched_track_ids = set()
        unmatched_boxes = []

        # Cố gắng khớp các box hiện tại với các track cũ
        for box in current_boxes:
            best_iou = 0
            best_track_id = -1
            for track_id, data in tracked_faces.items():
                iou = calculate_iou(box, data['box'])
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id != -1:
                tracked_faces[best_track_id]['box'] = box  # Cập nhật vị trí mới
                tracked_faces[best_track_id]['last_seen_frame'] = frame_counter
                matched_track_ids.add(best_track_id)
            else:
                unmatched_boxes.append(box)

        # Chạy nhận dạng cho các box mới xuất hiện (unmatched)
        if frame_counter % RECOGNITION_INTERVAL == 0 and unmatched_boxes:
            face_crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _ in unmatched_boxes]

            if face_crops:
                batch_input = np.array([preprocess_arcface(crop).squeeze() for crop in face_crops])
                raw_embeddings = arcface_engine(batch_input).reshape(len(face_crops), -1)

                for i, raw_embedding in enumerate(raw_embeddings):
                    embedding = raw_embedding / np.linalg.norm(raw_embedding)

                    best_match_info = None
                    if known_faces_cache:
                        similarities = [np.dot(embedding, known['embedding']) for known in known_faces_cache]
                        best_idx = np.argmax(similarities)
                        if similarities[best_idx] > COSINE_THRESHOLD:
                            best_match_info = known_faces_cache[best_idx]

                    # Tạo một track mới cho khuôn mặt này
                    tracked_faces[next_track_id] = {
                        "box": unmatched_boxes[i],
                        "name": best_match_info['name'] if best_match_info else "Unknown",
                        "user_id": best_match_info['id'] if best_match_info else None,
                        "last_seen_frame": frame_counter
                    }
                    next_track_id += 1

        for track_id, data in tracked_faces.items():
            if data['user_id'] is not None:
                user_id = data['user_id']
                if time.time() - last_check_in.get(user_id, 0) > CHECK_IN_COOLDOWN:
                    threading.Thread(target=check_in, args=(user_id,), daemon=True).start()
                    last_check_in[user_id] = time.time()

        # Xóa các track cũ không còn xuất hiện
        tracked_faces = {tid: data for tid, data in tracked_faces.items() if
                         frame_counter - data['last_seen_frame'] < RECOGNITION_INTERVAL * 2}

        # --- Vẽ kết quả lên frame ---
        for track_id, data in tracked_faces.items():
            x1, y1, x2, y2, _ = data['box']
            name = data['name']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Tính và hiển thị FPS
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1:
            fps = fps_frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            fps_frame_count = 0
            fps_start_time = time.time()

        latest_processed_frame = frame.copy()

    cap.release()
    is_running = False
    print("Vòng lặp AI đã kết thúc.")

# --- Chạy chương trình ---
if __name__ == "__main__":
    print("Khởi chạy server streaming...")
    streaming_thread = threading.Thread(target=run_streaming_server, daemon=True)
    streaming_thread.start()

    print("Khởi chạy vòng lặp xử lý AI...")
    try:
        ai_processing_loop()
    except KeyboardInterrupt:
        print("Nhận tín hiệu dừng (Ctrl+C). Đang tắt chương trình...")
        is_running = False
        time.sleep(1)  # Chờ các luồng con kết thúc
    finally:
        print("Chương trình đã tắt.")