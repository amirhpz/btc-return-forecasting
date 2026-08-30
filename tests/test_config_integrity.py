from pathlib import Path

from btc_forecasting.common.config import list_yaml_files, load_yaml
from btc_forecasting.experiments.registry import load_experiment_registry


def test_all_yaml_files_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    files = list_yaml_files(root)
    assert files
    for path in files:
        assert isinstance(load_yaml(path), dict)


def test_experiment_ids_are_unique_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_experiment_registry(root)
    assert list(sorted(registry)) == [f"E{number:02d}" for number in range(11)]
