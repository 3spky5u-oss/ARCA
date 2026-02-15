"""
Benchmark Harness v2
====================
Comprehensive chunking + retrieval pipeline optimization.

Layered architecture:
  L0: Chunking sweep (~90 configs)
  L1: Retrieval toggle sweep (~15 configs)
  L2: Continuous parameter sweep (OAT)
  L3: Answer generation via local LLM
  L4: LLM-as-judge scoring (configurable provider)
  L5: Statistical analysis + heatmaps
  L6: Failure categorization (configurable provider)
"""

__version__ = "2.1.0"
