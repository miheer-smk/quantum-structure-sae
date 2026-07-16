"""
Run logging — local JSONL logger + a global runs manifest.

Every experiment run gets a directory under `runs/` containing:
    config_resolved.yaml   — the exact config the run saw (incl. overrides)
    meta.json              — git commit, timestamp, seed record, env versions
    metrics.jsonl          — one JSON object per logged step/event
    results.json           — final results (written by `finish()`)

Every run also appends one line to `runs/manifest.jsonl`, mapping
(config, seed, git hash) → run directory, so any figure/table can be traced
back to the exact code + config + seed that produced it.

Backend: local JSONL by default. If Weights & Biases is installed and the
config sets `logging.wandb: true`, metrics are mirrored to W&B as well —
the local files are always written regardless, so reproducibility never
depends on an external service.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import save_config


def git_commit_hash(cwd: str | Path | None = None) -> str:
    """Current git commit hash ('unknown' outside a repo); '+dirty' if uncommitted changes."""
    try:
        root = str(cwd) if cwd else None
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=root, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=root, check=True,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class RunLogger:
    """
    Minimal experiment logger. One instance per run.

    Usage
    -----
        logger = RunLogger("smoke", cfg, seed_record)
        logger.log({"epoch": 1, "val_r2": 0.87})
        logger.finish({"test_r2": 0.99})
    """

    def __init__(
        self,
        run_name: str,
        cfg: dict[str, Any],
        seed_record: dict[str, Any] | None = None,
        runs_root: str | Path = "runs",
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(runs_root) / f"{run_name}_{stamp}_s{cfg.get('seed', 'NA')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._t0 = time.time()

        save_config(cfg, self.run_dir / "config_resolved.yaml")

        import numpy
        import torch
        self.meta = {
            "run_name": run_name,
            "run_dir": str(self.run_dir),
            "git_commit": git_commit_hash(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_record": seed_record or {"seed": cfg.get("seed")},
            "config_path": cfg.get("_config_path"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

        self._wandb = None
        if (cfg.get("logging") or {}).get("wandb"):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=cfg["logging"].get("wandb_project", "quantum-structure-sae"),
                    name=self.run_dir.name, config=cfg,
                )
            except ImportError:
                self.log({"event": "wandb_requested_but_not_installed"})

    def log(self, metrics: dict[str, Any]) -> None:
        """Append one metrics record (adds elapsed wall-clock seconds)."""
        rec = {"t": round(time.time() - self._t0, 3), **metrics}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(rec, default=_json_fallback) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics)

    def finish(self, results: dict[str, Any]) -> Path:
        """Write final results and append this run to runs/manifest.jsonl."""
        (self.run_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=_json_fallback)
        )
        manifest = self.run_dir.parent / "manifest.jsonl"
        entry = {
            "run_dir": str(self.run_dir),
            "run_name": self.meta["run_name"],
            "config_path": self.meta["config_path"],
            "seed": self.cfg.get("seed"),
            "git_commit": self.meta["git_commit"],
            "started_utc": self.meta["started_utc"],
            "wall_seconds": round(time.time() - self._t0, 1),
        }
        with open(manifest, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if self._wandb is not None:
            self._wandb.summary.update(results)
            self._wandb.finish()
        return self.run_dir


def _json_fallback(obj: Any):
    """Serialize numpy scalars/arrays that sneak into metrics dicts."""
    import numpy as np
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")
