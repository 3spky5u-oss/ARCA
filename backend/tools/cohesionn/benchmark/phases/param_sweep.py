"""
Phase 5: Parameter Sweep
========================
Sweep RAG parameters via runtime_config.update().

Each parameter is swept independently (not combinatorial).
Original config is saved and restored after each sweep.
"""

import time
from typing import Any, Dict, List

from .base import BasePhase, PhaseResult


class ParamSweepPhase(BasePhase):
    """Sweep RAG parameters one at a time via runtime_config."""

    phase_name = "param_sweep"

    def run(self) -> List[PhaseResult]:
        self._begin()
        results: List[PhaseResult] = []

        from config import runtime_config

        # Save original config
        original = runtime_config.to_dict()
        self._log(f"Saved original config ({len(original)} params)")

        sweep_params = self.config.sweep_params
        total_combos = sum(len(vals) for vals in sweep_params.values())
        self._log(f"Sweeping {len(sweep_params)} params ({total_combos} total values)")

        combo_idx = 0
        for param_name, values in sweep_params.items():
            self._log(f"\nSweeping {param_name} ({len(values)} values)...")
            param_results: List[PhaseResult] = []

            for val in values:
                combo_idx += 1
                variant_name = f"{param_name}={val}"
                self._log(f"  [{combo_idx}/{total_combos}] {variant_name}")
                t0 = time.time()

                try:
                    # Update the single parameter
                    runtime_config.update(**{param_name: val})

                    # Run retrieval with current config
                    retrieval = self._retrieve_all(rerank=True)

                    # Score
                    aggregate, per_query = self._score_all(variant_name, retrieval)
                    duration = time.time() - t0

                    result = self._make_result(
                        variant_name=variant_name,
                        model_spec=None,
                        aggregate=aggregate,
                        per_query=per_query,
                        duration_s=duration,
                        metadata={
                            "param": param_name,
                            "value": val,
                        },
                    )
                    param_results.append(result)
                    self._log(
                        f"    composite={aggregate.avg_composite:.3f}  "
                        f"keywords={aggregate.avg_keyword_hits:.1%}  "
                        f"{duration:.1f}s"
                    )

                except Exception as e:
                    self._log(f"    ERROR: {e}")
                    param_results.append(PhaseResult(
                        phase=self.phase_name,
                        variant_name=variant_name,
                        error=str(e),
                        duration_s=time.time() - t0,
                        metadata={"param": param_name, "value": val},
                    ))
                finally:
                    # Restore original value for this param
                    if param_name in original:
                        runtime_config.update(**{param_name: original[param_name]})

            results.extend(param_results)

        # Final restore of all params
        for key, val in original.items():
            try:
                runtime_config.update(**{key: val})
            except Exception:
                pass
        self._log("Restored original config")

        return results
