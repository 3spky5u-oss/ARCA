"""
Embedding worker subprocess — isolated CUDA context for ONNX embedder.

Spawned via multiprocessing.Process(target=worker_main, start_method="spawn").
Must use "spawn" — "fork" copies the parent's CUDA state and corrupts it.

If ONNX hits a CUDA error (cudaErrorIllegalAddress, cudaFreeHost abort),
the worker dies but the parent (uvicorn) survives and respawns it.
This is the same isolation pattern used by Ollama, vLLM, and llama-server.

IPC: multiprocessing.Queue with pickle. Each request/response carries a UUID
"id" for correlation. Overhead: ~1ms per round-trip vs ~50ms per embedding
batch — negligible.
"""

import logging
import os
import traceback


def worker_main(request_queue, response_queue, force_cpu, device_id, log_level):
    """
    Subprocess entry point. Loads ONNX model in its own CUDA context
    and processes embed requests via Queue IPC.
    """
    # Configure child process logging
    logging.basicConfig(
        level=log_level,
        format="[embed-worker] %(levelname)s %(message)s",
    )
    log = logging.getLogger("embed_worker")

    # Isolate CUDA device if specified
    if device_id is not None and not force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    embedder = None
    try:
        from tools.cohesionn.embeddings import UniversalEmbedder, EMBEDDING_DEVICE

        device = "cpu" if force_cpu else EMBEDDING_DEVICE
        embedder = UniversalEmbedder(device=device, device_id=device_id or 0)
        # Trigger model load
        dim = embedder.dimension
        log.info(
            f"Embedding worker started (PID {os.getpid()}, "
            f"device {device if device == 'cpu' else f'cuda:{device_id or 0}'}, "
            f"dim={dim})"
        )
    except Exception:
        log.error(f"Failed to load embedder:\n{traceback.format_exc()}")
        # Signal failure for dimension request — will raise in proxy
        try:
            req = request_queue.get(timeout=5)
            response_queue.put({
                "id": req.get("id", "init"),
                "error": f"Worker init failed: {traceback.format_exc()}",
            })
        except Exception:
            pass
        return

    # Main request loop
    while True:
        try:
            request = request_queue.get(timeout=30)
        except Exception:
            # Timeout — no requests, keep alive
            continue

        req_id = request.get("id", "unknown")
        req_type = request.get("type", "unknown")

        try:
            if req_type == "embed_query":
                vector = embedder.embed_query(request["text"])
                response_queue.put({"id": req_id, "vector": vector})

            elif req_type == "embed_documents":
                vectors = embedder.embed_documents(
                    request["texts"],
                    batch_size=request.get("batch_size", 64),
                )
                response_queue.put({"id": req_id, "vectors": vectors})

            elif req_type == "embed_queries":
                vectors = embedder.embed_queries(
                    request["texts"],
                    batch_size=request.get("batch_size", 64),
                )
                response_queue.put({"id": req_id, "vectors": vectors})

            elif req_type == "dimension":
                response_queue.put({"id": req_id, "dimension": embedder.dimension})

            elif req_type == "health":
                response_queue.put({"id": req_id, "healthy": True})

            elif req_type == "shutdown":
                log.info("Shutdown requested, exiting")
                response_queue.put({"id": req_id, "shutdown": True})
                break

            else:
                response_queue.put({
                    "id": req_id,
                    "error": f"Unknown request type: {req_type}",
                })

        except Exception:
            tb = traceback.format_exc()
            log.error(f"Request {req_type} failed:\n{tb}")
            response_queue.put({"id": req_id, "error": tb})

    # Cleanup
    if embedder is not None:
        try:
            embedder.unload()
        except Exception:
            pass
    log.info("Embedding worker exiting")
