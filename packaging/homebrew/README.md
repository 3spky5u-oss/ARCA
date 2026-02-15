# Homebrew Packaging

ARCA is distributed through Docker images; the Homebrew package should install only the `arca` local operator CLI wrapper.

## Recommended Structure

1. Maintain a dedicated tap repo, e.g. `3spky5u-oss/homebrew-arca`.
2. Publish an `arca` formula in the tap from release assets.
3. Keep this repository as the formula source template only.

## Install Flow (target)

```bash
brew tap 3spky5u-oss/arca
brew install arca
arca bootstrap
```

## Formula Template

Use `packaging/homebrew/arca.rb.template` and replace:
- `__VERSION__`
- `__URL__`
- `__SHA256__`
