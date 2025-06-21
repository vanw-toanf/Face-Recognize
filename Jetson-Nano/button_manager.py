import RPi.GPIO as GPIO
import time
import subprocess
import os
import signal

# --- Cấu hình chân GPIO ---
LED_PIN = 12
BUTTON_PIN = 18

# --- Biến trạng thái toàn cục ---
process = None  # Biến để lưu tiến trình của jetson_app.py
is_running = False


def setup_gpio():
    """Thiết lập chế độ và các chân GPIO."""
    GPIO.setmode(GPIO.BOARD)  # Sử dụng chế độ đánh số BOARD
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    # Thiết lập chân nút bấm với điện trở kéo lên nội tại
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print("GPIO đã được thiết lập.")


def start_face_recognition():
    """Bật đèn LED và khởi chạy script nhận diện."""
    global process, is_running
    if not is_running:
        print("Đang khởi động ứng dụng nhận diện...")
        GPIO.output(LED_PIN, GPIO.HIGH)  # Bật đèn LED
        # Sử dụng os.setsid để tạo một process group mới
        # -> có thể tắt toàn bộ tiến trình con khi cần
        process = subprocess.Popen(['./start_app.sh'], cwd=os.getcwd(), preexec_fn=os.setsid)
        is_running = True
        print(f"Ứng dụng đã khởi động với PID: {process.pid}")


def stop_face_recognition():
    """Tắt đèn LED và dừng script nhận diện."""
    global process, is_running
    if is_running and process:
        print(f"Đang dừng ứng dụng nhận diện (PID: {process.pid})...")
        GPIO.output(LED_PIN, GPIO.LOW)  # Tắt đèn LED
        # Gửi tín hiệu SIGTERM đến toàn bộ process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate
        process.wait()  # Chờ tiến trình kết thúc
        process = None
        is_running = False
        print("Ứng dụng đã dừng.")


def button_callback(channel):
    """Hàm được gọi khi có sự kiện nhấn nút."""
    # Chống rung phím (debounce) - đợi để tín hiệu ổn định
    time.sleep(0.1)
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        if is_running:
            stop_face_recognition()
        else:
            start_face_recognition()


def main():
    """Hàm chính của chương trình."""
    setup_gpio()

    # Thêm sự kiện ngắt cho chân nút bấm
    # Bắt sự kiện khi có sườn xuống (từ HIGH xuống LOW khi nhấn nút)
    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)

    print("Script quản lý đã sẵn sàng. Nhấn nút để Bật/Tắt.")
    print("Nhấn CTRL+C để thoát script quản lý.")

    try:
        # Giữ cho chương trình chạy mãi mãi để lắng nghe sự kiện
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Đang dọn dẹp GPIO...")
    finally:
        stop_face_recognition()  # tắt ứng dụng con khi thoát
        GPIO.cleanup()


if __name__ == '__main__':
    main()