# FaceSync backend — Python + FastAPI + dlib/face_recognition
# dlib is compiled from source, so the build needs cmake + a C++ toolchain.
# First build on Render will take ~10-15 minutes; later builds are cached by Render.

FROM python:3.11-slim

# --- System build dependencies for dlib / opencv ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Increase pip timeout — dlib source build is slow to download/compile.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
