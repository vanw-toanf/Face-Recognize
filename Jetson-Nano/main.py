import threading
import time
from ai_processing.main_loop import ai_processing_loop
import streaming_server

if __name__ == "__main__":
    print("Khởi chạy server streaming...")
    streaming_thread = threading.Thread(target=streaming_server.run_in_thread, daemon=True)
    streaming_thread.start()

    print("Khởi chạy vòng lặp xử lý AI...")
    try:
        # Chạy vòng lặp AI ở luồng chính
        ai_processing_loop()
    except KeyboardInterrupt:
        print("\nNhận tín hiệu dừng (Ctrl+C). Đang tắt chương trình...")
    finally:
        # Báo cho các luồng con biết để dừng lại
        streaming_server.is_running = False
        # Chờ luồng streaming kết thúc
        time.sleep(1)
        print("Chương trình đã tắt.")