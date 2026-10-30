"""
One training run: a condition (stain-handling method), a held-out center (fold) and a seed.

The run checks the freeze, trains on the fold's ``train`` tiles with the fixed WILDS schedule
(DR-011), then — in evaluation mode, without augmentation — saves for every tile of every role:
the two logits, P(tumour) and the 2,048-d penultimate features (what the monitor's Tier 2 reads).
It writes everything to its own run directory and refuses to overwrite a finished run.

Run directory contents:
    summary.json             settings, provenance (freeze hash, git state), headline numbers
    train_log.csv            loss, accuracy and time per epoch
    predictions_<role>.csv   one row per tile: ids, label, logits, p_tumour
    features_<role>.npy      float16 (n_tiles, 2048), rows aligned with predictions_<role>.csv
    model.pt                 final weights
    reference_tiles.csv      E2/E3 only: the training tiles the stain reference was fitted on
    normalizer.npz           E2/E3 only: the fitted reference parameters

Pre-registration lock: test roles (``test_in``, ``test_cross``) are only scored once
``docs/analysis_plan.md`` is committed to git with no uncommitted changes, so no test result can
be seen before the analysis plan is fixed (DR-012). Until then, use ``roles=("train",)``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stain_benchmark.data.dataset import ROLES, TileDataset, fold_table, make_loaders
from stain_benchmark.data.freeze import verify_freeze
from stain_benchmark.evaluation.metrics import compute_calibration_slope
from stain_benchmark.models.classifier.lightning_module import ResNet50Classifier
from stain_benchmark.models.dann.gradient_reversal import ScheduledGRL
from stain_benchmark.models.dann.lightning_module import DomainClassifier
from stain_benchmark.normalization.base import StainNormalizer
from stain_benchmark.normalization.macenko import MacenkoNormalizer
from stain_benchmark.normalization.reinhard import ReinhardNormalizer
from stain_benchmark.utils.reproducibility import (
    WorkerInitFn,
    compute_experiment_hash,
    make_deterministic,
    set_seed,
)

CONDITIONS = ("E1", "E2", "E3", "E5")
DANN_GAMMA, DANN_MAX_LAMBDA, DANN_DOMAIN_WEIGHT = 10.0, 1.0, 1.0   # Ganin et al. 2016 (DR-017)
NORMALIZERS = {"E2": MacenkoNormalizer, "E3": ReinhardNormalizer}
N_REFERENCE = 200       # tiles the stain reference is fitted on (DR-013)
REFERENCE_SEED = 0      # fixed, so every training seed of a fold shares one reference
REPO = Path(__file__).resolve().parents[3]
ANALYSIS_PLAN = "docs/analysis_plan.md"


class PreRegistrationError(RuntimeError):
    """Raised when test sets would be scored before the analysis plan is committed."""


def require_committed_plan(repo: Path = REPO, plan: str = ANALYSIS_PLAN) -> str:
    """Return the commit that last changed the analysis plan; raise unless it is committed and clean."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()
    last = git("log", "-1", "--format=%H", "--", plan)
    if not (repo / plan).exists() or not last:
        raise PreRegistrationError(f"{plan} is not committed: test sets stay sealed until it is")
    if git("status", "--porcelain", "--", plan):
        raise PreRegistrationError(f"{plan} has uncommitted changes: commit them before scoring test sets")
    return last


@dataclass(frozen=True)
class Schedule:
    """WILDS Camelyon17 settings (Koh et al., ICML 2021), fixed in advance (DR-011)."""

    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 10


WILDS = Schedule()


class GlobalMeanPool(nn.Module):
    """Spatial mean, identical to ResNet's adaptive average pool to (1, 1) but with a
    deterministic GPU backward pass (adaptive_avg_pool2d's backward is not deterministic)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


def build_model(pretrained: bool = True) -> ResNet50Classifier:
    """ResNet-50 (ImageNet weights) with a dropout + linear head, shared by every condition."""
    model = ResNet50Classifier(num_classes=2, pretrained=pretrained, dropout=0.5)
    model.backbone.avgpool = GlobalMeanPool()
    return model


def fit_reference(condition: str, tiles_dir: Path, table: pd.DataFrame,
                  n: int = N_REFERENCE) -> tuple[StainNormalizer | None, pd.DataFrame]:
    """Stain normaliser for ``condition`` fitted on ``n`` training tiles of this fold (DR-013).

    Only ``train`` rows are eligible, so the held-out centre never informs the reference. The
    sampled tiles are stacked into one tall image and the normaliser is fitted on it. E1 has none.
    """
    if condition not in NORMALIZERS:
        return None, table.iloc[:0]
    train = table[table["role"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(REFERENCE_SEED)
    ref = train.iloc[np.sort(rng.choice(len(train), min(n, len(train)), replace=False))]
    mosaic = np.concatenate([np.array(Image.open(tiles_dir / p).convert("RGB")) for p in ref["path"]], axis=0)
    return NORMALIZERS[condition]().fit(mosaic), ref


def _git_state(repo: Path) -> tuple[str, bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=repo, capture_output=True, text=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


@torch.no_grad()
def predict(model: ResNet50Classifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Logits (n, 2) and penultimate features (n, 2048) for every tile, in loader order."""
    model.eval()
    logits, feats = [], []
    for batch in loader:
        f = model.get_features(batch["images"].to(device, non_blocking=True))
        logits.append(model.backbone.fc(f).float().cpu())
        feats.append(f.half().cpu())
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def _role_metrics(pred: pd.DataFrame) -> dict:
    y, p = pred["label"].to_numpy(), pred["p_tumour"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"n_tiles": len(y), "auroc": None, "calibration_slope": None}
    slope, _ = compute_calibration_slope(y, p)
    return {"n_tiles": len(y), "n_patients": int(pred["patient_id"].nunique()),
            "n_tumour_patients": int(pred.loc[pred.label == 1, "patient_id"].nunique()),
            "auroc": float(roc_auc_score(y, p)), "calibration_slope": slope}


def train_run(tiles_dir: str | Path, holdout: str | int, seed: int, run_dir: str | Path,
              condition: str = "E1", schedule: Schedule = WILDS, num_workers: int = 8,
              device: str | None = None, pretrained: bool = True, verify: bool = True,
              progress: bool = True, roles: tuple[str, ...] = ROLES, require_plan: bool = True) -> dict:
    """Train and evaluate one (condition, fold, seed); return the summary written to summary.json.

    ``roles`` are the tile sets scored after training. Scoring a test role requires the committed
    analysis plan (``require_plan``; switched off only in unit tests on synthetic data).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} not implemented; available: {CONDITIONS}")
    plan_commit = None
    if require_plan and any(r.startswith("test") for r in roles):
        plan_commit = require_committed_plan()
    tiles_dir, run_dir = Path(tiles_dir), Path(run_dir)
    if (run_dir / "summary.json").exists():
        raise FileExistsError(f"{run_dir} already holds a finished run; use a new directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    make_deterministic()                       # sets CUBLAS_WORKSPACE_CONFIG (ISS-026)
    set_seed(seed, deterministic=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if verify:
        verify_freeze(tiles_dir)               # before any tile is read
    normalizer, ref = fit_reference(condition, tiles_dir, fold_table(tiles_dir, holdout))
    if normalizer is not None:
        ref[["tile_id", "patient_id", "center", "label"]].to_csv(run_dir / "reference_tiles.csv", index=False)
        normalizer.save(run_dir / "normalizer.npz")
    norm_fn = normalizer.transform if normalizer is not None else None
    loaders, table = make_loaders(tiles_dir, holdout, seed, batch_size=schedule.batch_size,
                                  num_workers=num_workers, normalizer=norm_fn, verify=False)
    model = build_model(pretrained).to(dev)
    params = list(model.parameters())
    dann = condition == "E5"
    if dann:
        # E5: domain-adversarial training over the TRAINING centres only (DR-008); the held-out
        # centre is never seen. Same tumour head as E1; the domain branch only shapes the features.
        n_domains = int(table.loc[table["role"] == "train", "center"].nunique())
        domain_head = DomainClassifier(input_dim=2048, hidden_dim=1024, dropout=0.5,
                                       num_domains=n_domains).to(dev)
        grl = ScheduledGRL(gamma=DANN_GAMMA, max_lambda=DANN_MAX_LAMBDA)
        params += list(domain_head.parameters())
        total_steps = schedule.epochs * len(loaders["train"])
    opt = torch.optim.SGD(params, lr=schedule.lr, momentum=schedule.momentum,
                          weight_decay=schedule.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    step = 0

    log, t_start = [], time.time()
    for epoch in range(1, schedule.epochs + 1):
        model.train()
        if dann:
            domain_head.train()
        t0, total, correct, loss_sum, dom_correct = time.time(), 0, 0, 0.0, 0
        bar = tqdm(loaders["train"], desc=f"{condition} fold {holdout} seed {seed} · epoch {epoch}/{schedule.epochs}",
                   unit="batch", disable=not progress)
        for batch in bar:
            x = batch["images"].to(dev, non_blocking=True)
            y = batch["labels"].to(dev, non_blocking=True)
            if dann:
                lam = grl.update_lambda(step / total_steps)
                f = model.get_features(x)
                out = model.backbone.fc(f)
                dom_out = domain_head(grl(f))
                d = batch["domain"].to(dev, non_blocking=True)
                loss = loss_fn(out, y) + DANN_DOMAIN_WEIGHT * loss_fn(dom_out, d)
                dom_correct += (dom_out.argmax(1) == d).sum().item()
            else:
                out = model(x)
                loss = loss_fn(out, y)
            step += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y); total += len(y)
            correct += (out.argmax(1) == y).sum().item()
            bar.set_postfix(loss=f"{loss_sum / total:.4f}", acc=f"{correct / total:.3f}")
        log.append({"epoch": epoch, "loss": loss_sum / total, "accuracy": correct / total,
                    "seconds": round(time.time() - t0, 1),
                    **({"domain_accuracy": dom_correct / total, "lambda": round(lam, 4)} if dann else {})})
        print(f"epoch {epoch}: loss {loss_sum / total:.4f} · train accuracy {correct / total:.3f} "
              f"· {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    if dann:
        torch.save(domain_head.state_dict(), run_dir / "domain_head.pt")

    metrics = {}
    for role in roles:
        sub = table[table["role"] == role].reset_index(drop=True)
        loader = DataLoader(TileDataset(tiles_dir, sub, normalizer=norm_fn, augment=False),
                            batch_size=128, shuffle=False,
                            num_workers=num_workers, worker_init_fn=WorkerInitFn(seed),
                            pin_memory=dev.type == "cuda")
        logits, feats = predict(model, loader, dev)
        p = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
        pred = sub[["tile_id", "patient_id", "slide_id", "center", "label"]].assign(
            logit_0=logits[:, 0], logit_1=logits[:, 1], p_tumour=p)
        pred.to_csv(run_dir / f"predictions_{role}.csv", index=False)
        np.save(run_dir / f"features_{role}.npy", feats)
        metrics[role] = _role_metrics(pred)

    a_in = metrics.get("test_in", {}).get("auroc")
    a_cross = metrics.get("test_cross", {}).get("auroc")
    freeze = json.loads((tiles_dir / "freeze.json").read_text()) if (tiles_dir / "freeze.json").exists() else {}
    head, dirty = _git_state(Path(__file__).resolve().parents[3])
    settings = {"condition": condition, "holdout": str(holdout), "seed": seed, "schedule": asdict(schedule),
                "pretrained": pretrained, "freeze_master_sha256": freeze.get("master_sha256")}
    summary = {
        **settings,
        "experiment_hash": compute_experiment_hash(settings, seed),
        "roles_scored": list(roles), "analysis_plan_commit": plan_commit,
        "metrics": metrics,
        "delta_g_auroc": (a_in - a_cross) if None not in (a_in, a_cross) else None,
        "train_seconds": round(time.time() - t_start, 1),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "git_head": head, "git_dirty": dirty,
        "note": "Point estimates only; patient-clustered intervals are computed in the analysis step.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
