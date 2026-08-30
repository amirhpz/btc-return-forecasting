$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv is not installed. Install it with: winget install --id=astral-sh.uv -e"
}

uv python install 3.12
uv sync
uv run btc-forecast doctor
uv run btc-forecast validate-configs
uv run pytest
