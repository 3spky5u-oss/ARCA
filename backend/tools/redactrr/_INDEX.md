# backend/tools/redactrr/

PII detection and document redaction tool. Uses a 3-phase detection pipeline (regex scan, heuristic name detection, LLM single-pass) and supports both PDF and DOCX output with proper text removal (not visual overlay).

| File | Purpose | Key Exports |
|------|---------|-------------|
| __init__.py | Public API | `Redactrr`, `redact_document()` |
| __main__.py | `python -m tools.redactrr` CLI entry | -- |
| redactor.py | Core: PIIDetector (3-phase), WordRedactor, PDFRedactor, main Redactrr pipeline | `Redactrr`, `PIIDetector`, `PIIType`, `RedactionResult`, `WordRedactor`, `PDFRedactor` |
| cli.py | CLI with `redact`, `preview`, `batch` subcommands | `main()` |
