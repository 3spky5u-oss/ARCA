# backend/services/

Infrastructure singletons providing database, cache, LLM, and external service connectivity. Every service follows the pattern: try connection, graceful fallback, health check. All use lazy initialization via `get_*()` factory functions with asyncio locks.

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Module exports | `RedisManager`, `get_redis` |
| admin_auth.py | Multi-user auth: PostgreSQL + file fallback, bcrypt, JWT HS256 (8h expiry) | `AdminAuthManager`, `verify_admin`, `verify_user` |
| hardware.py | GPU/CPU/RAM detection, VRAM budget calculator, self-calibrating LLM timeout | `HardwareProfile`, `get_hardware_profile()`, `VRAMBudget` |
| database.py | AsyncPG pool with SQLite fallback, auto-schema from init.sql | `get_database()`, `DatabaseManager` |
| redis_client.py | Async Redis with OrderedDict LRU fallback (1000 max, TTL support) | `get_redis()`, `RedisManager` |
| neo4j_client.py | Neo4j driver, dynamic schema (auto-constraints for new entity types) | `get_neo4j()`, `Neo4jClient` |
| llm_server_manager.py | llama-server lifecycle, VRAM budget check, startup timeout | `get_server_manager()`, `LLMServerManager` |
| llm_config.py | ModelSlot definitions (chat, vision), env-var-configurable ports/models | `ModelSlot`, `get_model_slots()` |
| llm_client.py | OpenAI SDK wrapper for llama-server, streaming think-tag extraction | `LLMClient`, `get_llm_client()` |
| json_repair.py | 8-step JSON repair pipeline for malformed LLM output | `parse_json_response()` |
| system_metrics.py | CPU/RAM/Disk/GPU metrics via psutil + nvidia-smi | `get_system_metrics()` |
| session_cache.py | Redis-backed ChatSession persistence with 24h TTL | `SessionCache` |
| phii_cache.py | Redis-backed Phii profile persistence (energy, specialty, expertise) | `PhiiCache` |
