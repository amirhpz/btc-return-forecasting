from __future__ import annotations

import json

from btc_forecasting.common.config import find_project_root
from btc_forecasting.training.lstm_horizon_walkforward import (
    run_lstm_horizon_walkforward_experiment,
)


def main() -> None:
    root = find_project_root()
    result = run_lstm_horizon_walkforward_experiment(project_root=root)
    print(json.dumps(result.result, indent=2, sort_keys=True))
    print(f"artifact={result.run_directory.relative_to(root)}")


if __name__ == "__main__":
    main()
