#!/bin/bash
# ARCA - Startup Script
# LLM inference is handled by llama-server (llama.cpp) inside the backend container.
# Models are GGUF files in the ./models/ directory.

echo "=========================================="
echo "  ARCA - Your Junior Engineer"
echo "=========================================="

echo ""
echo "Starting Docker services..."
docker compose --compatibility up -d

echo ""
echo "Waiting for backend to be ready..."
until curl -sf http://localhost:8000/health &>/dev/null; do
    echo "  Still waiting for backend..."
    sleep 2
done
echo "  Backend is ready!"

echo ""
echo "=========================================="
echo "  ARCA is ready!"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "=========================================="
