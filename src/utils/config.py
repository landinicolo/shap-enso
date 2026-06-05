"""Config loading, merging, and resolution utilities."""

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config file and optionally merge a flat dict of dot-path overrides.

    Dot-path keys like "model.xgb.max_depth" are supported in overrides so CLI
    args can override individual fields without rewriting the whole config.

    Args:
        path: Path to a YAML config file.
        overrides: Flat dict of dot-separated key paths → values.

    Returns:
        Merged config dict with all paths resolved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    return resolve_paths(cfg)


def _apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Apply flat dot-path overrides onto a nested config dict (mutates a copy)."""
    cfg = copy.deepcopy(cfg)
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def resolve_paths(cfg: dict) -> dict:
    """Expand ~ and $ENV_VAR in every string value that looks like a path."""
    cfg = copy.deepcopy(cfg)
    _walk_and_resolve(cfg)
    return cfg


def _walk_and_resolve(node: Any) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and ("/" in v or v.startswith("~")):
                node[k] = os.path.expandvars(os.path.expanduser(v))
            else:
                _walk_and_resolve(v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str) and ("/" in v or v.startswith("~")):
                node[i] = os.path.expandvars(os.path.expanduser(v))
            else:
                _walk_and_resolve(v)


def get_lead_configs(cfg: dict) -> list[dict]:
    """Expand a multi-lead config into one config dict per lead month.

    Each returned dict has cfg['data']['lead_months'] replaced with a single int
    and cfg['experiment']['name'] suffixed with '_lead{L}'.
    """
    leads = cfg["data"]["lead_months"]
    configs = []
    for lead in leads:
        c = copy.deepcopy(cfg)
        c["data"]["lead_months"] = [lead]
        c["experiment"]["name"] = f"{cfg['experiment']['name']}_lead{lead}"
        configs.append(c)
    return configs


def get_model_output_dir(cfg: dict, lead: int) -> Path:
    """Return the directory where model artifacts for a given lead are stored."""
    model_type = cfg["model"]["type"]
    task = cfg["model"]["task"]
    base = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / model_type
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_shap_output_dir(cfg: dict) -> Path:
    """Return the SHAP output directory, creating it if needed."""
    d = Path(cfg["shap"]["output_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d
