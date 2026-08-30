from pathlib import Path

from btc_forecasting.reporting.manifest import create_run_manifest


def test_manifest_contains_reproducibility_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = create_run_manifest(project_root=root, experiment_id="E00", run_id="test")
    assert manifest["experiment_id"] == "E00"
    assert "python" in manifest
    assert "platform" in manifest
    assert "git" in manifest
