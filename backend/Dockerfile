# FaceSync backend — Python + FastAPI + dlib/face_recognition
# dlib is compiled from source, so the build needs cmake + a C++ toolchain.
# First build on Render will take ~10-15 minutes; later builds are cached by Render.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir dlib-bin==19.24.6

RUN pip install --no-cache-dir --no-deps face_recognition==1.3.0

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
