"""Smoke test for Phase 0 infrastructure: config loading and logging."""

import sys
from pathlib import Path

# Make src importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_lead_configs, get_model_output_dir, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

REPO = Path(__file__).parent.parent
DEFAULT_CFG = REPO / "configs" / "default.yaml"


def test_load_default():
    cfg = load_config(DEFAULT_CFG)
    assert cfg["experiment"]["seed"] == 42
    assert cfg["data"]["lead_months"] == [3, 6, 12]
    assert cfg["model"]["type"] == "xgboost"
    log.info("default.yaml loaded OK")


def test_dot_path_override():
    cfg = load_config(DEFAULT_CFG, overrides={"model.type": "lstm", "model.xgb.max_depth": 6})
    assert cfg["model"]["type"] == "lstm"
    assert cfg["model"]["xgb"]["max_depth"] == 6
    log.info("dot-path override OK")


def test_get_lead_configs():
    cfg = load_config(DEFAULT_CFG)
    lead_cfgs = get_lead_configs(cfg)
    assert len(lead_cfgs) == 3
    assert lead_cfgs[0]["data"]["lead_months"] == [3]
    assert lead_cfgs[2]["experiment"]["name"].endswith("_lead12")
    log.info("get_lead_configs OK — %d configs", len(lead_cfgs))


def test_experiment_configs():
    for name in [
        "xgb_regression_all_leads",
        "xgb_classification_all_leads",
        "lstm_regression_all_leads",
        "cnn_regression_all_leads",
    ]:
        path = REPO / "configs" / f"{name}.yaml"
        cfg = load_config(path)
        assert "experiment" in cfg
        assert "model" in cfg
        log.info("  %s: model.type=%s, model.task=%s", name, cfg["model"]["type"], cfg["model"]["task"])
    log.info("all experiment configs loaded OK")


def test_path_resolution():
    cfg = load_config(DEFAULT_CFG)
    raw = cfg["data"]["raw_dir"]
    assert not raw.startswith("~"), f"~ not expanded: {raw}"
    assert "$" not in raw, f"env var not expanded: {raw}"
    log.info("path resolution OK: raw_dir=%s", raw)


if __name__ == "__main__":
    tests = [
        test_load_default,
        test_dot_path_override,
        test_get_lead_configs,
        test_experiment_configs,
        test_path_resolution,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            log.error("FAIL %s: %s", t.__name__, e)
            failed.append(t.__name__)

    if failed:
        log.error("%d test(s) failed: %s", len(failed), failed)
        sys.exit(1)
    else:
        log.info("All Phase 0 tests passed.")
