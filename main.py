import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from insightface.app import FaceAnalysis
from utils import load_registered_faces, recognize_face
import time
import requests

LOG_SERVER_URL = "http://10.42.0.1:8000/log"

# --- TensorRT Setup ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # Copy input data to device
    [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Copy output data to host
    [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]
    stream.synchronize()
    return [out["host"] for out in outputs]

# Load TensorRT engine and create execution context
engine = load_engine("models/best.engine")
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Prepare InsightFace recognizer
face_recognizer = FaceAnalysis(name='buffalo_l')
face_recognizer.prepare(ctx_id=0)

# Load registered faces
known_faces = load_registered_faces()

# Open webcam
cap = cv2.VideoCapture(0)

last_seen = {}
start_time = time.time()

print("Start face attendance with TensorRT engine...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame to input format expected by your model
    # (Bạn cần tùy chỉnh theo model YOLO bạn convert sang engine)
    input_img = cv2.resize(frame, (640, 640))  # Giả sử model nhận 640x640
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.ascontiguousarray(input_img)

    # Copy input data to host buffer
    np.copyto(inputs[0]["host"], input_img.ravel())

    # Run inference
    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
    output = trt_outputs[0]  # output shape, cần decode

    # TODO: decode output tensor thành boxes (x1, y1, x2, y2, conf, class_id)
    # Cách decode output của YOLO TensorRT engine tùy theo model convert
    # Mình sẽ giả sử bạn đã có hàm decode_output(output) trả về danh sách box:

    boxes = decode_output(output, frame.shape[1], frame.shape[0])  # bạn tự viết hàm này

    for box in boxes[:5]:  # Giới hạn 5 mặt đầu tiên
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        face_crop = frame[y1:y2, x1:x2]

        name = recognize_face(face_crop, face_recognizer, known_faces)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name != "Unknown":
            now_time = time.time()
            if name not in last_seen or now_time - last_seen[name] > 5:
                now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    requests.post(LOG_SERVER_URL, json={"name": name, "timestamp": now_str})
                    print(f"📤 Gửi log thành công: {name}")
                except Exception as e:
                    print(f"❌ Lỗi gửi log: {e}")

                last_seen[name] = now_time

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    start_time = end_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Attendance TensorRT", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
