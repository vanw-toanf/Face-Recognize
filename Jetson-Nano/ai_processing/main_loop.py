import cv2
import time
import numpy as np
import threading
import config
import server_connector
import streaming_server
from .engine import TRT_Engine
from . import processing

def ai_processing_loop():
    """Vòng lặp chính cho việc phát hiện, nhận dạng và xử lý."""
    print("Đang tải model...")
    yolo = TRT_Engine(config.YOLO_ENGINE_PATH, max_batch_size=1)
    arcface = TRT_Engine(config.RECOGNIZER_ENGINE_PATH, max_batch_size=config.MAX_FACES_PER_FRAME)
    print("Tải model thành công!")

    # Khởi tạo camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera."); streaming_server.is_running = False; return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # Các biến trạng thái
    frame_counter = 0
    tracked_faces = {}
    next_track_id = 0
    last_sync_time = 0
    last_check_in = {}
    fps_start_time, fps_frame_count = time.time(), 0

    while streaming_server.is_running:
        # Đồng bộ với server định kỳ
        if time.time() - last_sync_time > config.SYNC_INTERVAL_SECONDS:
            server_connector.sync_with_server()
            last_sync_time = time.time()

        ret, frame = cap.read()
        if not ret: break
        frame_counter += 1

        # Phát hiện khuôn mặt
        yolo_input, scale, dx, dy = processing.preprocess_yolo(frame)
        processing.postprocess_yolo.scale_params = (scale, dx, dy) # Truyền tham số cho hàm postprocess
        yolo_output = yolo(yolo_input)
        current_boxes = processing.postprocess_yolo(yolo_output, frame.shape[:2])
        current_boxes = sorted(current_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:config.MAX_FACES_PER_FRAME]

        # Logic Tracking và Nhận dạng
        matched_track_ids, unmatched_boxes = set(), []
        for box in current_boxes:
            best_iou, best_track_id = 0, -1
            for track_id, data in tracked_faces.items():
                iou = processing.calculate_iou(box, data['box'])
                if iou > config.IOU_THRESHOLD and iou > best_iou:
                    best_iou, best_track_id = iou, track_id

            if best_track_id != -1:
                tracked_faces[best_track_id].update({'box': box, 'last_seen_frame': frame_counter})
                matched_track_ids.add(best_track_id)
            else:
                unmatched_boxes.append(box)

        # Chạy nhận dạng cho các khuôn mặt mới
        if frame_counter % config.RECOGNITION_INTERVAL == 0 and unmatched_boxes:
            face_crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _ in unmatched_boxes]
            if face_crops:
                batch_input = np.array([processing.preprocess_arcface(crop).squeeze() for crop in face_crops])
                raw_embeddings = arcface(batch_input).reshape(len(face_crops), -1)

                for i, raw_embedding in enumerate(raw_embeddings):
                    embedding = raw_embedding / np.linalg.norm(raw_embedding)
                    best_match = None
                    if server_connector.known_faces_cache:
                        sims = [np.dot(embedding, f['embedding']) for f in server_connector.known_faces_cache]
                        best_idx = np.argmax(sims)
                        if sims[best_idx] > config.COSINE_THRESHOLD:
                            best_match = server_connector.known_faces_cache[best_idx]

                    tracked_faces[next_track_id] = {
                        "box": unmatched_boxes[i],
                        "name": best_match['name'] if best_match else "Unknown",
                        "user_id": best_match['id'] if best_match else None,
                        "last_seen_frame": frame_counter
                    }
                    next_track_id += 1

        # Gửi yêu cầu check-in
        for data in tracked_faces.values():
            user_id = data.get('user_id')
            if user_id and time.time() - last_check_in.get(user_id, 0) > config.CHECK_IN_COOLDOWN_SECONDS:
                threading.Thread(target=server_connector.check_in, args=(user_id,), daemon=True).start()
                last_check_in[user_id] = time.time()

        # Xóa các track cũ
        tracked_faces = {tid: d for tid, d in tracked_faces.items() if frame_counter - d['last_seen_frame'] < config.RECOGNITION_INTERVAL * 2}

        # Vẽ kết quả và FPS lên frame
        for data in tracked_faces.values():
            x1, y1, x2, y2, _ = data['box']
            name = data['name']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_frame_count += 1
        if time.time() - fps_start_time > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            fps_frame_count, fps_start_time = 0, time.time()

        # Cập nhật frame cho luồng streaming
        streaming_server.latest_processed_frame = frame.copy()

    cap.release()
    print("Vòng lặp AI đã kết thúc.")