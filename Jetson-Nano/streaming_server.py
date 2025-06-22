from fastapi import FastAPI
from starlette.responses import StreamingResponse
import uvicorn
import asyncio
import cv2
import time
import config

app = FastAPI()
latest_processed_frame = None
is_running = True

def run_in_thread():
    """Chạy server uvicorn trong một luồng riêng."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    uvicorn_config = uvicorn.Config(app, host=config.STREAMING_HOST, port=config.STREAMING_PORT, log_level="warning")
    server = uvicorn.Server(uvicorn_config)
    server.run()

@app.get("/video_feed")
def video_feed():
    def generate():
        while is_running:
            if latest_processed_frame is not None:
                (flag, encodedImage) = cv2.imencode(".jpg", latest_processed_frame)
                if not flag: continue
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03) # Giảm tải CPU
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")