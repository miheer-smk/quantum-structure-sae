"""Fast unit tests for the Phase-0 reproducibility harness (config/repro/runlog)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from qsae.config import load_config, save_config
from qsae.repro import set_global_seed, seeded_generator
from qsae.runlog import RunLogger, git_commit_hash


@pytest.fixture()
def tiny_cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 7\ntrain:\n  lr: 3.0e-4\n  epochs: 10\n")
    return p


class TestConfig:
    def test_load_and_types(self, tiny_cfg):
        cfg = load_config(tiny_cfg)
        assert cfg["seed"] == 7
        assert cfg["train"]["lr"] == pytest.approx(3e-4)
        assert cfg["_config_path"] == str(tiny_cfg)

    def test_overrides_dotted_and_yaml_typed(self, tiny_cfg):
        cfg = load_config(tiny_cfg, overrides=["train.epochs=2", "seed=1", "new.flag=true"])
        assert cfg["train"]["epochs"] == 2
        assert cfg["seed"] == 1
        assert cfg["new"]["flag"] is True

    def test_missing_seed_rejected(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("train:\n  lr: 0.1\n")
        with pytest.raises(ValueError, match="seed"):
            load_config(p)

    def test_bad_override_rejected(self, tiny_cfg):
        with pytest.raises(ValueError, match="key=value"):
            load_config(tiny_cfg, overrides=["no_equals_sign"])

    def test_save_roundtrip(self, tiny_cfg, tmp_path):
        cfg = load_config(tiny_cfg)
        out = tmp_path / "resolved.yaml"
        save_config(cfg, out)
        cfg2 = load_config(out)
        cfg2["_config_path"] = cfg["_config_path"]
        assert cfg2 == cfg


class TestRepro:
    def test_seeding_reproduces_all_three_rngs(self):
        set_global_seed(123)
        a = (np.random.rand(3), torch.rand(3))
        set_global_seed(123)
        b = (np.random.rand(3), torch.rand(3))
        assert np.allclose(a[0], b[0])
        assert torch.allclose(a[1], b[1])

    def test_record_contents(self):
        rec = set_global_seed(5)
        assert rec["seed"] == 5
        assert rec["deterministic_requested"] is True

    def test_seeded_generator_independent_of_global(self):
        g1 = seeded_generator(9)
        np.random.rand(100)  # perturb global state
        g2 = seeded_generator(9)
        assert np.allclose(g1.uniform(size=5), g2.uniform(size=5))


class TestRunLogger:
    def test_end_to_end_artifacts(self, tiny_cfg, tmp_path):
        cfg = load_config(tiny_cfg)
        logger = RunLogger("t", cfg, {"seed": 7}, runs_root=tmp_path / "runs")
        logger.log({"step": 1, "loss": np.float32(0.5)})  # numpy scalar must serialize
        run_dir = logger.finish({"final": 1.0, "arr": np.arange(3)})

        assert (run_dir / "config_resolved.yaml").exists()
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["seed_record"]["seed"] == 7
        assert meta["git_commit"]  # captured (may be 'unknown' outside a repo)

        lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert json.loads(lines[0])["loss"] == 0.5

        results = json.loads((run_dir / "results.json").read_text())
        assert results["arr"] == [0, 1, 2]

        manifest = (tmp_path / "runs" / "manifest.jsonl").read_text().strip().splitlines()
        entry = json.loads(manifest[-1])
        assert entry["run_dir"] == str(run_dir)
        assert entry["seed"] == 7

    def test_git_hash_format(self):
        h = git_commit_hash()
        assert h == "unknown" or len(h.split("+")[0]) == 40
