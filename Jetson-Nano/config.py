# --- CẤU HÌNH ---

# Cấu hình Server
SERVER_IP = "10.42.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

# Cấu hình Model
YOLO_ENGINE_PATH = "models/detect/best.engine"
RECOGNIZER_ENGINE_PATH = "models/recognize/model-r34.engine"
YOLO_INPUT_SHAPE = (320, 320)  # (h, w)
RECOGNIZER_INPUT_SHAPE = (112, 112)  # (h, w)

# Cấu hình xử lý
CONF_THRESHOLD = 0.5        # Ngưỡng tin cậy cho YOLO
NMS_THRESHOLD = 0.4         # Ngưỡng cho Non-Maximum Suppression
COSINE_THRESHOLD = 0.30     # Ngưỡng nhận dạng (cần tinh chỉnh)
IOU_THRESHOLD = 0.4         # Ngưỡng IoU để tracking
RECOGNITION_INTERVAL = 10   # Số frame giữa 2 lần nhận dạng
MAX_FACES_PER_FRAME = 5     # Số lượng khuôn mặt xử lý tối đa trong 1 frame

# Cấu hình Camera và Streaming
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
STREAMING_HOST = "0.0.0.0"
STREAMING_PORT = 8001

# Cấu hình Giao tiếp
SYNC_INTERVAL_SECONDS = 60
CHECK_IN_COOLDOWN_SECONDS = 10