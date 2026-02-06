FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for insightface (Cython), opencv, onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Cython first (insightface build dependency)
RUN pip install --no-cache-dir cython numpy

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download InsightFace buffalo_l model pack during build (not at runtime)
RUN python -c "from insightface.app import FaceAnalysis; fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); fa.prepare(ctx_id=-1)"

# Copy app code
COPY . .

# Railway sets PORT dynamically — must not hardcode
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
