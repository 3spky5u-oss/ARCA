# Stage 1: Build llama-server with CUDA support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# CUDA compute capabilities: 75=Turing(RTX20xx), 80=Ampere(A100),
# 86=Ampere(RTX30xx), 89=Ada(RTX40xx), 90=Hopper(H100), 120=Blackwell(RTX50xx)
ARG CUDA_ARCHS="75;80;86;89"

RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp (pinned commit for reproducible + cacheable builds)
# Update this SHA to upgrade llama.cpp: git log --oneline -1
ARG LLAMA_CPP_SHA=39bf692af1cb
RUN git clone https://github.com/ggml-org/llama.cpp.git /build/llama.cpp && \
    cd /build/llama.cpp && git checkout ${LLAMA_CPP_SHA}
WORKDIR /build/llama.cpp

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_BACKEND_DL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build build --target llama-server -j$(nproc)

# Collect all runtime artifacts into a single directory
RUN mkdir -p /app/lib && \
    cp build/bin/llama-server /app/ && \
    find build/bin -name "*.so*" -exec cp {} /app/lib/ \; && \
    cp /usr/local/cuda/lib64/libcudart.so* /app/lib/ && \
    cp /usr/local/cuda/lib64/libcublas.so* /app/lib/ && \
    cp /usr/local/cuda/lib64/libcublasLt.so* /app/lib/

# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
# - libgomp1: needed by ggml-cpu backend
# - tesseract-ocr: OCR engine for Docling
# - libxcb1, libgl1, libglib2.0-0, libsm6: headless rendering for Docling layout engine
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    tesseract-ocr \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies FIRST — before builder artifacts.
# This layer caches as long as requirements.txt doesn't change, regardless of
# whether the llama.cpp builder stage rebuilds. BuildKit cache mount keeps pip's
# download cache (~3GB of wheels) across layer invalidations too.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 -r requirements.txt && \
    pip install --force-reinstall --no-deps onnxruntime-gpu && \
    python -m spacy download en_core_web_sm

# Register ONNX Runtime's CUDA/cuDNN libs for the linker
# (torch installs these to site-packages/nvidia/*/lib/ but onnxruntime can't find them)
RUN find /usr/local/lib/python3.12/site-packages/nvidia -name "lib" -type d > /etc/ld.so.conf.d/nvidia-pip.conf

# Copy llama-server binary and all shared libs from builder
COPY --from=builder /app/llama-server /usr/local/bin/llama-server
COPY --from=builder /app/lib/ /usr/local/lib/

# Backend plugins must be next to binary for GGML_BACKEND_DL discovery
RUN ln -sf /usr/local/lib/libggml-cuda.so /usr/local/bin/ && \
    ln -sf /usr/local/lib/libggml-cpu.so /usr/local/bin/
RUN ldconfig

# Copy application code
COPY . .

# Create non-root user + install gosu for entrypoint privilege drop
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/entrypoint.sh

EXPOSE 8000

# Entrypoint runs as root to fix volume ownership, then drops to appuser
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "1048576"]
