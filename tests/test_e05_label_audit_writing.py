from __future__ import annotations

import json
from pathlib import Path

from btc_forecasting.targets.trend_label_audit import _write_summary_artifact


def test_trend_audit_writes_final_json_into_new_directory(tmp_path: Path) -> None:
    final_path = tmp_path / "outputs" / "data" / "trend_label_audit" / "summary.json"
    payload: dict[str, object] = {
        "audit_id": "E05-L-A",
        "label_definitions_unchanged": True,
    }

    assert not final_path.parent.exists()
    _write_summary_artifact(final_path, payload)

    assert final_path.is_file()
    assert final_path.name == "summary.json"
    assert json.loads(final_path.read_text(encoding="utf-8")) == payload
    assert not list(final_path.parent.glob("*.tmp"))
    assert not list(final_path.parent.glob("*.tmp.tmp"))
