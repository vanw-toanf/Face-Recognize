#!/bin/bash

# Dòng này đảm bảo script chạy từ đúng thư mục gốc của người dùng
cd "$(dirname "$0")"

# Kích hoạt môi trường ảo
source ../../env/bin/activate

# Chạy ứng dụng nhận diện với LD_PRELOAD
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 jetson_app.py