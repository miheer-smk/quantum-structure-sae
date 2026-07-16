"""
Config system — YAML experiment configs with dotted-key overrides.

Every experiment is driven by a versioned YAML file under `configs/`, with an
explicit `seed` key. No magic numbers in experiment code — they live here.

Usage
-----
    cfg = load_config("configs/smoke.yaml", overrides=["train.epochs=2", "seed=1"])
    cfg["train"]["epochs"]   # plain nested dict
    save_config(cfg, run_dir / "config_resolved.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load a YAML config file and apply dotted-key overrides.

    Parameters
    ----------
    path      : path to a YAML file. Must contain a top-level `seed` key
                (reproducibility is non-negotiable).
    overrides : optional list of "dotted.key=value" strings (values parsed
                as YAML, so "lr=3e-4" gives a float and "flag=true" a bool).

    Returns
    -------
    cfg : nested plain dict. `cfg["_config_path"]` records the source file.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    if "seed" not in cfg:
        raise ValueError(f"{path} must define a top-level `seed` key")

    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not of the form key=value")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))

    cfg["_config_path"] = str(path)
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg["a"]["b"] = value for dotted == "a.b", creating levels as needed."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {dotted!r}: {k!r} is not a mapping")
    node[keys[-1]] = value


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write the resolved config next to a run's outputs (YAML, sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
