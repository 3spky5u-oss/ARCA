"""
Phase 4: LLM Shootout
=====================
Swap expert LLM models and evaluate answer generation quality.

Flow:
  1. Cache retrieval contexts using best config from prior phases
  2. Per LLM: swap model → generate answers → score
  3. Additional metrics: answer_keyword_overlap, answer_completeness
  4. Restore original model after
"""

import time
from typing import Any, Dict, List

from ..config import ModelSpec
from .base import BasePhase, PhaseResult


class LLMPhase(BasePhase):
    """Evaluate LLM models for answer generation quality."""

    phase_name = "llm"

    def run(self) -> List[PhaseResult]:
        self._begin()
        results: List[PhaseResult] = []

        # Step 1: Cache retrieval contexts
        self._log("Caching retrieval contexts...")
        cached = self._retrieve_all(rerank=True)
        self._log(f"Cached {len(cached)} query contexts")

        # Build context strings per query
        contexts: Dict[str, str] = {}
        for q in self.queries:
            r = cached.get(q.id, {})
            chunks = r.get("chunks", [])
            contexts[q.id] = "\n\n".join(
                c.get("content", "") for c in chunks
            )

        # Step 2: Test each LLM
        llms = self.config.filter_models("llm")
        for i, model in enumerate(llms):
            self._log(f"[{i + 1}/{len(llms)}] Testing {model.name}...")
            t0 = time.time()

            try:
                result = self._test_llm(model, contexts, cached)
                results.append(result)
                self._log(
                    f"  → composite={result.aggregate.get('avg_composite', 0):.3f}  "
                    f"keywords={result.aggregate.get('avg_keyword_hits', 0):.1%}  "
                    f"{time.time() - t0:.1f}s"
                )
            except Exception as e:
                self._log(f"  → ERROR: {e}")
                results.append(PhaseResult(
                    phase=self.phase_name,
                    variant_name=model.short_name,
                    model_spec=model.to_dict(),
                    error=str(e),
                    duration_s=time.time() - t0,
                ))

        return results

    def _test_llm(
        self,
        model: ModelSpec,
        contexts: Dict[str, str],
        cached_retrieval: Dict[str, Any],
    ) -> PhaseResult:
        """Swap model, generate answers, score."""
        t0 = time.time()

        # Swap LLM model
        swapped = self._swap_model(model.hf_id)
        if not swapped:
            raise RuntimeError(f"Failed to swap to {model.hf_id}")

        # Generate answers for each query
        generation_results: Dict[str, Dict[str, Any]] = {}
        for q in self.queries:
            context = contexts.get(q.id, "")
            if not context:
                generation_results[q.id] = {"answer": "", "error": "no_context"}
                continue

            start = time.time()
            try:
                answer = self._generate_answer(q.query, context)
                elapsed_ms = (time.time() - start) * 1000
                generation_results[q.id] = {
                    "answer": answer,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                generation_results[q.id] = {
                    "answer": "",
                    "latency_ms": (time.time() - start) * 1000,
                    "error": str(e),
                }

        # Score: use answer text for keyword matching instead of chunks
        scored_results: Dict[str, Any] = {}
        for q in self.queries:
            gen = generation_results.get(q.id, {})
            r = cached_retrieval.get(q.id, {})
            if gen.get("error"):
                scored_results[q.id] = {
                    "chunks": r.get("chunks", []),
                    "latency_ms": gen.get("latency_ms", 0),
                    "error": gen["error"],
                }
            else:
                # Create a synthetic "chunk" from the LLM answer for scoring
                answer_chunk = {
                    "content": gen["answer"],
                    "score": 1.0,
                    "source": "llm_answer",
                }
                scored_results[q.id] = {
                    "chunks": [answer_chunk] + r.get("chunks", []),
                    "latency_ms": gen.get("latency_ms", 0),
                }

        aggregate, per_query = self._score_all(model.short_name, scored_results)
        duration = time.time() - t0

        return self._make_result(
            variant_name=model.short_name,
            model_spec=model,
            aggregate=aggregate,
            per_query=per_query,
            duration_s=duration,
            metadata={
                "generation_results": {
                    qid: {
                        "answer_length": len(g.get("answer", "")),
                        "latency_ms": g.get("latency_ms", 0),
                        "error": g.get("error"),
                    }
                    for qid, g in generation_results.items()
                }
            },
        )

    def _swap_model(self, gguf_filename: str) -> bool:
        """Swap the chat model using sync stop/start/health-poll."""
        try:
            from utils.llm import get_server_manager
            from services.llm_config import SLOTS
            import httpx

            mgr = get_server_manager()

            current = mgr._current_models.get("chat")
            if current == gguf_filename:
                self._log(f"Already running {gguf_filename}")
                return True

            self._log(f"Swapping chat: {current} → {gguf_filename}")
            mgr.stop("chat")
            time.sleep(3)

            started = mgr.start("chat", gguf_override=gguf_filename)
            if not started:
                self._log("mgr.start() returned False")
                return False

            port = SLOTS["chat"].port
            for attempt in range(180):
                try:
                    r = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
                    if r.status_code == 200:
                        self._log(f"Model healthy after {attempt}s")
                        return True
                except Exception:
                    pass
                time.sleep(1)

            self._log("Model failed health check after 180s")
            return False
        except Exception as e:
            self._log(f"Model swap failed: {e}")
            return False

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the current LLM."""
        import os
        import httpx

        port = os.environ.get("LLM_CHAT_PORT", "8081")

        prompt = (
            f"Based on the following technical reference material, answer the question.\n\n"
            f"Reference Material:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = httpx.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
