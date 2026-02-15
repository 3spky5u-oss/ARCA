# Winget Packaging

Winget should install ARCA's local operator wrapper, then users run `arca bootstrap`.

## Recommended Workflow

1. Create/install package manifests in a dedicated packaging repo.
2. Point installer to a release asset containing:
   - `scripts/arca.py`
   - wrapper launcher for Windows (`arca.cmd` or packaged executable)
3. Submit/update manifests in `microsoft/winget-pkgs`.

## Target UX

```powershell
winget install 3spky5u.ARCA
arca bootstrap
```

This repo keeps only documentation/templates for now.
