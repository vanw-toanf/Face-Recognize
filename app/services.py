import onnxruntime
import cv2
import numpy as np
from typing import List

recognize_model = "./exp/recognize/model-r34.onnx"
ort_session = onnxruntime.InferenceSession(recognize_model)

def get_embedding_from_image(image_bytes: bytes) -> List[float]:
    """
    Trích xuất embedding từ file ảnh (dạng bytes) bằng model ONNX.
    """
    global ort_session
    try:
        input_name = ort_session.get_inputs()[0].name

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ảnh khuôn mặt được crop (chỉ chứa khuôn mặt)
        img_resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, axis=0)

        ort_inputs = {input_name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)

        embedding = ort_outs[0].flatten()
        normalized_embedding = embedding / np.linalg.norm(embedding)

        return normalized_embedding.tolist()
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding bằng ONNX: {e}")
        return None