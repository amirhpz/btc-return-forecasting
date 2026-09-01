from pathlib import Path

import yaml


AUDIT_PATH = Path("configs/features/eng52_audit.yaml")
VALID_STATUSES = {"SAFE", "SAFE_WITH_ADAPTATION", "REJECT"}
MANDATORY_FIELDS = {
    "feature_name",
    "source_file",
    "documented_definition_formula",
    "required_raw_inputs",
    "required_historical_lookback",
    "causal_at_decision_time",
    "valid_on_1h_ohlcv",
    "requires_gap_safe_adaptation",
    "overlap_with_f0",
    "leakage_or_data_availability_risk",
    "status",
}


def test_eng52_audit_is_complete_and_unique() -> None:
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))
    features = audit["features"]

    assert len(features) == 52
    names = [feature["feature_name"] for feature in features]
    assert len(set(names)) == 52
    assert all(MANDATORY_FIELDS <= feature.keys() for feature in features)
    assert all(feature["status"] in VALID_STATUSES for feature in features)

    actual_counts = {
        status: sum(feature["status"] == status for feature in features)
        for status in VALID_STATUSES
    }
    assert audit["summary_counts"] == actual_counts
