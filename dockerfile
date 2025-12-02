# 1. Use the FULL Python image (bookworm)
# This includes a more complete Linux OS than 'slim', solving many missing library issues.
FROM python:3.11-bookworm

# 2. Install ALL system dependencies required for Computer Vision
# This list fixes the "libGL", "cv2", and "ImportError" crashes.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Basic build tools
    build-essential \
    # Graphics libraries required by OpenCV
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Cleanup to save space
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install Python libraries DIRECTLY (No requirements.txt needed)
# We upgrade pip first.
RUN pip install --no-cache-dir --upgrade pip

# We force install websockets and numpy first to prevent build errors.
RUN pip install --no-cache-dir websockets numpy

# We use --prefer-binary to force downloading pre-compiled versions of OpenCV and YOLO
# This prevents the container from trying to compile C++ code and failing.
RUN pip install --no-cache-dir --prefer-binary opencv-python-headless ultralytics

# 5. Copy YOUR existing, working code and model
# Ensure these files are in the folder when you run build!
COPY adas_websocket_server.py .
COPY yolov8n.pt .

# 6. Expose the port
EXPOSE 8765

# 7. Run the server
CMD ["python", "adas_websocket_server.py"]