"""Tests for one training run (CPU, untrained weights, one short epoch)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from stain_benchmark.training.run import (
    GlobalMeanPool,
    PreRegistrationError,
    Schedule,
    build_model,
    require_committed_plan,
    train_run,
)

TINY = Schedule(epochs=1, batch_size=4)


def _run(frozen, run_dir, seed=42):
    return train_run(frozen, holdout=1, seed=seed, run_dir=run_dir, schedule=TINY,
                     num_workers=0, device="cpu", pretrained=False, progress=False, require_plan=False)


def test_global_mean_pool_equals_adaptive_avg_pool():
    x = torch.randn(2, 8, 5, 7)
    assert torch.allclose(GlobalMeanPool()(x), nn.AdaptiveAvgPool2d((1, 1))(x))


def test_model_uses_deterministic_pool():
    assert isinstance(build_model(pretrained=False).backbone.avgpool, GlobalMeanPool)


def test_run_writes_all_outputs(frozen, tmp_path):
    s = _run(frozen, tmp_path / "run")
    run = tmp_path / "run"
    for name in ("summary.json", "train_log.csv", "model.pt"):
        assert (run / name).exists()
    # holdout 1: centers 0 and 2 train on 2 patients each; val + test patients form test_in.
    table = {"train": 8, "test_in": 8, "test_cross": 8}       # 2 tiles per patient
    for role, n in table.items():
        pred = pd.read_csv(run / f"predictions_{role}.csv")
        feats = np.load(run / f"features_{role}.npy")
        assert len(pred) == n and feats.shape == (n, 2048) and feats.dtype == np.float16
        assert np.allclose(pred.p_tumour, 1 / (1 + np.exp(pred.logit_0 - pred.logit_1)), atol=1e-5)
    assert set(pd.read_csv(run / "predictions_test_cross.csv").center) == {1}
    saved = json.loads((run / "summary.json").read_text())
    assert saved["experiment_hash"] == s["experiment_hash"]
    assert saved["schedule"]["lr"] == 1e-3 and saved["freeze_master_sha256"]
    assert s["delta_g_auroc"] == pytest.approx(
        s["metrics"]["test_in"]["auroc"] - s["metrics"]["test_cross"]["auroc"])


def test_refuses_to_overwrite(frozen, tmp_path):
    _run(frozen, tmp_path / "run")
    with pytest.raises(FileExistsError):
        _run(frozen, tmp_path / "run")


def test_same_seed_same_predictions(frozen, tmp_path):
    _run(frozen, tmp_path / "a", seed=7)
    _run(frozen, tmp_path / "b", seed=7)
    _run(frozen, tmp_path / "c", seed=8)
    a, b, c = (pd.read_csv(tmp_path / r / "predictions_test_cross.csv") for r in "abc")
    assert np.array_equal(a.logit_1.to_numpy(), b.logit_1.to_numpy())
    assert not np.array_equal(a.logit_1.to_numpy(), c.logit_1.to_numpy())


def test_unknown_condition_raises(frozen, tmp_path):
    with pytest.raises(ValueError, match="not implemented"):
        train_run(frozen, 1, 42, tmp_path / "x", condition="E9", device="cpu", pretrained=False)


def test_test_sets_sealed_without_committed_plan(frozen, tmp_path):
    with pytest.raises(PreRegistrationError):
        train_run(frozen, 1, 42, tmp_path / "x", schedule=TINY, num_workers=0, device="cpu",
                  pretrained=False, progress=False)
    assert not (tmp_path / "x" / "model.pt").exists()        # refused before any training


def test_train_only_run_needs_no_plan(frozen, tmp_path):
    s = train_run(frozen, 1, 42, tmp_path / "x", schedule=TINY, num_workers=0, device="cpu",
                  pretrained=False, progress=False, roles=("train",))
    assert s["roles_scored"] == ["train"] and s["delta_g_auroc"] is None
    assert not (tmp_path / "x" / "predictions_test_in.csv").exists()


def test_plan_lock_on_a_git_repo(tmp_path):
    import subprocess
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    with pytest.raises(PreRegistrationError, match="not committed"):
        require_committed_plan(tmp_path)
    (tmp_path / "docs").mkdir(); (tmp_path / "docs" / "analysis_plan.md").write_text("plan\n")
    with pytest.raises(PreRegistrationError, match="not committed"):
        require_committed_plan(tmp_path)
    git("add", "docs/analysis_plan.md"); git("commit", "-qm", "plan")
    assert len(require_committed_plan(tmp_path)) == 40
    (tmp_path / "docs" / "analysis_plan.md").write_text("changed\n")
    with pytest.raises(PreRegistrationError, match="uncommitted"):
        require_committed_plan(tmp_path)


@pytest.mark.parametrize("condition", ["E2", "E3"])
def test_stain_reference_from_training_tiles_only(frozen, tmp_path, condition):
    for seed in (1, 2):
        train_run(frozen, holdout=1, seed=seed, run_dir=tmp_path / f"{condition}_{seed}", condition=condition,
                  schedule=TINY, num_workers=0, device="cpu", pretrained=False, progress=False,
                  require_plan=False)
    ref1 = pd.read_csv(tmp_path / f"{condition}_1" / "reference_tiles.csv")
    ref2 = pd.read_csv(tmp_path / f"{condition}_2" / "reference_tiles.csv")
    assert 1 not in set(ref1.center)                             # never the held-out centre
    assert ref1.tile_id.tolist() == ref2.tile_id.tolist()        # same reference for every seed
    assert (tmp_path / f"{condition}_1" / "normalizer.npz").exists()


def test_normalised_condition_changes_inputs(frozen, tmp_path):
    from stain_benchmark.data.dataset import fold_table
    from stain_benchmark.training.run import fit_reference
    table = fold_table(frozen, holdout=1)
    norm, ref = fit_reference("E3", frozen, table)
    assert norm is not None and len(ref) == (table.role == "train").sum()     # fewer than 200 available
    assert fit_reference("E1", frozen, table)[0] is None


def test_e5_dann_uses_training_centres_only(frozen, tmp_path):
    s = train_run(frozen, holdout=1, seed=3, run_dir=tmp_path / "e5", condition="E5",
                  schedule=Schedule(epochs=2, batch_size=4), num_workers=0, device="cpu",
                  pretrained=False, progress=False, require_plan=False)
    run = tmp_path / "e5"
    log = pd.read_csv(run / "train_log.csv")
    assert {"domain_accuracy", "lambda"} <= set(log.columns)
    assert 0 <= log.domain_accuracy.min() and log.domain_accuracy.max() <= 1
    assert log["lambda"].iloc[0] < log["lambda"].iloc[-1] < 1          # reversal strength ramps up
    head = torch.load(run / "domain_head.pt")
    assert head["classifier.8.weight"].shape[0] == 2                   # centres 0 and 2 only
    assert len(pd.read_csv(run / "predictions_test_cross.csv")) == 8
    assert s["condition"] == "E5"


def test_loaders_give_domain_only_for_training_tiles(frozen):
    from stain_benchmark.data.dataset import make_loaders
    loaders, _ = make_loaders(frozen, holdout=1, seed=1, batch_size=8, num_workers=0)
    batch = next(iter(loaders["train"]))
    assert set(batch["domain"].tolist()) <= {0, 1}
    assert "domain" not in next(iter(loaders["test_cross"]))
