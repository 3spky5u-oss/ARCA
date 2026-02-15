# Contributing to ARCA

## Getting Started

```bash
git clone https://github.com/3spky5u-oss/ARCA.git
cd ARCA
python scripts/arca.py bootstrap
```

Frontend: http://localhost:3000
Backend API: http://localhost:8000
Admin panel: http://localhost:3000/admin

## Project Structure

```
backend/           # FastAPI backend + llama-server LLM inference
frontend/          # Next.js 16 frontend
domains/           # Domain packs (tools, lexicon, branding)
docs/              # Documentation
scripts/           # Startup, utility, and audit scripts
```

## Development Workflow

**Backend changes:**
```bash
docker compose up -d --build backend
docker compose logs -f backend
```

**Frontend changes:**
```bash
cd frontend
npm install
npm run dev    # Dev server on :3000 with hot reload
```

**Run tests:**
```bash
docker compose exec backend pytest tests/ -v
```

**Useful helper commands:**
```bash
python scripts/arca.py doctor   # preflight checks
python scripts/arca.py update   # pull + restart
python scripts/arca.py down     # stop services
```

## Code Style

- **Python**: Follow existing patterns. Type hints where they add clarity. No unnecessary docstrings.
- **TypeScript/React**: Functional components. Tailwind for styling. No CSS modules.
- **Comments**: Explain *why*, not *what*. If the code needs a comment to explain what it does, the code should be clearer.
- **Commits**: Concise message, imperative mood. One logical change per commit.

## Domain Packs

The main way to extend ARCA is through domain packs. See [docs/dev/domain_packs.md](docs/dev/domain_packs.md) for the full guide.

A domain pack adds:
- Custom tools and executors
- Domain-specific vocabulary and detection patterns
- Personality and branding
- Additional API routes

## Areas for Contribution

- **Core platform improvements** — RAG pipeline, admin panel, chat engine
- **New domain packs** — Specialize ARCA for your industry
- **Hardware compatibility** — Testing on different GPU/CPU configurations
- **Documentation** — Guides, examples, translations
- **Bug reports** — File issues with reproduction steps

## Submitting Changes

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes
4. Test locally (`docker compose up -d --build`)
5. Submit a pull request with a clear description

## Reporting Issues

File issues at https://github.com/3spky5u-oss/ARCA/issues with:
- What you expected vs what happened
- Steps to reproduce
- Your hardware (GPU, RAM, OS)
- Backend logs if relevant (`docker compose logs backend`)

## License

By contributing, you agree that your contributions will be licensed under the same [MIT license](LICENSE) as the project.
