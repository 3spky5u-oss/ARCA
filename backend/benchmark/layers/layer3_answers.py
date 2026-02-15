"""
Layer 3: Answer Generation

Uses the optimal pipeline config from L0+L1+L2 to generate answers
for all benchmark queries via the local LLM.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class AnswerGenerationLayer(BaseLayer):
    """Layer 3: Generate answers using optimal retrieval + local LLM."""

    LAYER_NAME = "layer3_answers"

    def execute(self, result: LayerResult) -> LayerResult:
        from benchmark.collection_manager import BenchmarkCollectionManager
        from benchmark.config import ChunkingConfig
        from tools.cohesionn.retriever import CohesionnRetriever
        from config import runtime_config
        import httpx
        import os

        # Load optimal configs from previous layers
        optimal_l2 = self.load_optimal_config("layer2_params")
        if not optimal_l2:
            # Fall back to L1 if L2 hasn't run
            optimal_l2 = {
                "chunking": self.load_optimal_config("layer0_chunking"),
                "retrieval": self.load_optimal_config("layer1_retrieval"),
                "params": {},
            }

        chunking = optimal_l2.get("chunking", {})
        retrieval = optimal_l2.get("retrieval", {})
        optimal_params = optimal_l2.get("params", {})

        if not chunking:
            result.errors.append("No optimal chunking config found")
            return result

        chunk_cfg = ChunkingConfig.from_dict(chunking)
        retrieval_toggles = retrieval.get("toggles", {}) if retrieval else {}

        # Apply optimal params to runtime config
        if optimal_params:
            runtime_config.update(**optimal_params)
            logger.info(f"Applied optimal params: {optimal_params}")

        # Load queries
        queries_path = Path(self.config.output_dir) / "layer0_chunking" / "queries.json"
        from benchmark.queries.battery import BenchmarkQuery
        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        queries = [BenchmarkQuery.from_dict(q) for q in raw_queries]

        result.configs_total = len(queries)

        # Ingest corpus
        collection_mgr = BenchmarkCollectionManager()
        l0_corpus_dir = Path(self.config.output_dir) / "layer0_chunking" / "corpus_md"
        md_files = sorted(str(f) for f in l0_corpus_dir.glob("*.md"))

        ingest_config_id = f"l3_{chunk_cfg.config_id}"
        topic = collection_mgr.get_topic(ingest_config_id)

        n_chunks = collection_mgr.ingest_corpus(
            config_id=ingest_config_id,
            md_files=md_files,
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            context_prefix=chunk_cfg.context_prefix,
        )
        logger.info(f"Ingested {n_chunks} chunks for answer generation")

        # Generate answers
        retriever = CohesionnRetriever()
        llm_port = os.environ.get("LLM_CHAT_PORT", "8081")
        llm_url = f"http://localhost:{llm_port}/v1/chat/completions"
        answers = []

        for q in queries:
            if self.checkpoint.is_completed(self.LAYER_NAME, q.id):
                result.configs_skipped += 1
                saved = self.checkpoint.get_result(self.LAYER_NAME, q.id)
                if saved:
                    answers.append(saved)
                continue

            try:
                # Retrieve context
                retrieval_result = retriever.retrieve(
                    query=q.query,
                    topics=[topic],
                    top_k=self.config.top_k,
                    **retrieval_toggles,
                )
                context = retrieval_result.get_context(max_chunks=5)

                # Generate answer via local LLM
                answer_text = ""
                try:
                    llm_prompt = (
                        f"Based on the following context, answer the question.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {q.query}\n\n"
                        f"Answer:"
                    )

                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(
                            llm_url,
                            json={
                                "model": runtime_config.model_chat,
                                "messages": [{"role": "user", "content": llm_prompt}],
                                "max_tokens": runtime_config.max_output_tokens,
                                "temperature": 0.3,
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            answer_text = data["choices"][0]["message"]["content"]
                        else:
                            logger.warning(f"LLM returned {resp.status_code} for {q.id}")
                            answer_text = f"[LLM error: {resp.status_code}]"
                except Exception as e:
                    logger.warning(f"LLM call failed for {q.id}: {e}")
                    answer_text = f"[LLM error: {e}]"

                answer_data = {
                    "query_id": q.id,
                    "query": q.query,
                    "tier": q.tier,
                    "context": context,
                    "answer": answer_text,
                    "ground_truth": q.ground_truth_answer,
                    "source_projects": q.source_projects,
                    "n_chunks_retrieved": len(retrieval_result.chunks),
                    "max_score": retrieval_result.max_score,
                    "confidence": retrieval_result.confidence,
                }

                answers.append(answer_data)
                self.checkpoint.mark_completed(self.LAYER_NAME, q.id, answer_data)
                result.configs_completed += 1

                logger.info(f"  {q.id}: {len(answer_text)} chars, {retrieval_result.confidence} confidence")

            except Exception as e:
                result.errors.append(f"{q.id}: {e}")
                logger.error(f"Answer generation failed for {q.id}: {e}")

        # Cleanup
        collection_mgr.cleanup_topic(ingest_config_id)

        # Save all answers
        answers_path = self.output_dir / "answers.json"
        answers_path.write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")

        result.summary = {
            "total_queries": len(queries),
            "answers_generated": result.configs_completed,
            "avg_context_confidence": "see answers.json",
        }

        return result
