$ErrorActionPreference = "Stop"

uv python install 3.12
uv sync --extra cu126
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
