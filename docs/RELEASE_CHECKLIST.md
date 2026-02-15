# Public Release Checklist

Use this checklist before pushing ARCA updates to public GitHub.

## 1. Repo Hygiene

- [ ] `git status` is clean for the intended public branch.
- [ ] `.env` is not tracked.
- [ ] `CLAUDE.md`, `TODO.md`, `Dev/`, private domains, and internal docs are not in the public branch.
- [ ] No runtime artifacts are tracked (`__pycache__`, reports, benchmarks, model files).

## 2. Security

- [ ] Secret scan passes on staged/tracked files.
- [ ] Docker/service credentials in docs are placeholders only.
- [ ] No internal URLs, tokens, or private hostnames appear in tracked docs/scripts.

## 3. Build + Startup

- [ ] `python scripts/arca.py doctor` passes (or only expected warnings).
- [ ] `docker compose pull` succeeds.
- [ ] `docker compose up -d` succeeds.
- [ ] `python scripts/smoke_test.py --frontend` passes.

## 4. Model Bootstrap

- [ ] `python scripts/model_bootstrap.py --check-only` behavior is correct when `./models` is empty.
- [ ] Backend startup auto-download works when `ARCA_AUTO_DOWNLOAD_MODELS=true` and `LLM_CHAT_MODEL_REPO` is set.
- [ ] Backend gracefully degrades to CPU when GPU/driver is unavailable (no crash loop).
- [ ] Bootstrap docs reflect current model-download flow.

## 5. Docs Accuracy

- [ ] `README.md` quick start matches actual commands.
- [ ] `docs/API.md` and smoke-test endpoint expectations match backend routes.
- [ ] `docs/SECURITY.md` matches current container/runtime behavior.
- [ ] `scripts/_INDEX.md` only lists files that exist.

## 6. Public Export

- [ ] If the public export stack is running, stop it first (`docker compose down` in export dir).
- [ ] Run `python scripts/export_public.py --clean`.
- [ ] Review `PUBLIC_EXPORT_REPORT.md` in export directory.
- [ ] Confirm no blocked/private files are present in export tree.
- [ ] Push export tree to GitHub remote.

## 7. Post-Release

- [ ] Tag release and publish notes.
- [ ] Verify container image publishing workflow completed (`.github/workflows/publish-images.yml`).
- [ ] Test fresh install path from public repo only.
