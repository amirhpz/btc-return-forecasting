from __future__ import annotations

import json

from btc_forecasting.common.config import find_project_root
from btc_forecasting.targets.trend_label_audit import run_trend_label_audit


def main() -> None:
    root = find_project_root()
    result = run_trend_label_audit(project_root=root)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"artifact={result.artifact_path.relative_to(root)}")


if __name__ == "__main__":
    main()
