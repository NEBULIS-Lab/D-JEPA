"""Measured-outcome supervision and provenance-preserving feature-cache training."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import RelationalAlignment, ordinal_ranks

SYNTHETIC_SCOPE = "synthetic_plumbing_test_not_robot_result"


def source_receipt(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "identity_check": "path/stat only; not a content hash"}


def load_feature_cache(path):
    with np.load(path, allow_pickle=False) as f:
        required = {"terminal", "goals", "candidate_ids", "native_costs", "feature_spec", "source_model"}
        if not required <= set(f.files):
            raise ValueError(f"missing cache fields: {sorted(required - set(f.files))}")
        cache = {name: f[name] for name in required}
    terminal, goals, costs, ids = (cache[k] for k in ("terminal", "goals", "native_costs", "candidate_ids"))
    if terminal.ndim != 4 or min(terminal.shape) < 1:
        raise ValueError("terminal must be nonempty [B,N,G,D]")
    b, n, g, d = terminal.shape
    if d < 2 or goals.shape != (b, g, d) or costs.shape != (b, n, g) or ids.shape != (b, n):
        raise ValueError("feature-cache shapes disagree")
    if any(x.dtype.kind != "f" or not np.isfinite(x).all() for x in (terminal, goals, costs)):
        raise ValueError("feature-cache tensors must be finite floating arrays")
    if ids.dtype.kind != "i":
        raise ValueError("candidate IDs must be signed integers")
    ordinal_ranks(torch.from_numpy(costs[:, :, 0].copy()), torch.from_numpy(ids.copy()))
    for key in ("feature_spec", "source_model"):
        if cache[key].shape != () or cache[key].dtype.kind not in "US" or not str(cache[key].item()).strip():
            raise ValueError(f"{key} must be an explicit nonempty scalar string")
        cache[key] = str(cache[key].item())
    cache["source"] = source_receipt(path)
    return cache


def load_training_cache(path, *, allow_synthetic=False):
    cache = load_feature_cache(path)
    with np.load(path, allow_pickle=False) as f:
        required = {"true_costs", "observed", "provenance", "evidence_ids", "start_ids", "split"}
        if not required <= set(f.files):
            raise ValueError(f"missing supervision fields: {sorted(required - set(f.files))}")
        cache.update({key: f[key] for key in required})
    b, n = cache["candidate_ids"].shape
    for key in ("true_costs", "observed", "provenance", "evidence_ids"):
        if cache[key].shape != (b, n):
            raise ValueError(f"{key} must be [B,N]")
    if cache["observed"].dtype != np.bool_ or not cache["observed"].all():
        raise ValueError("all training outcomes must be observed; unknown candidates are not labels")
    if not np.isfinite(cache["true_costs"]).all():
        raise ValueError("true costs must be finite measured costs (lower is better)")
    kinds = set(cache["provenance"].astype(str).ravel())
    if "synthetic_test" in kinds:
        if not allow_synthetic or kinds != {"synthetic_test"}:
            raise ValueError("synthetic fixtures require explicit opt-in and cannot mix with measured outcomes")
        scope = SYNTHETIC_SCOPE
    elif not kinds <= {"executed_real", "executed_sim"}:
        raise ValueError(f"unknown/unexecuted provenance cannot supervise training: {sorted(kinds)}")
    else:
        scope = "measured_outcome_training_not_formal_evaluation"
    if cache["evidence_ids"].dtype.kind not in "US" or np.any(np.char.str_len(cache["evidence_ids"]) == 0):
        raise ValueError("every measured candidate needs a nonempty evidence reference")
    for key in ("start_ids", "split"):
        if cache[key].shape != (b,) or cache[key].dtype.kind not in "US" or np.any(cache[key] == ""):
            raise ValueError(f"{key} must contain a nonempty string per decision")
    if set(cache["split"]) != {"train", "val"}:
        raise ValueError("both train and val decisions required, no other split names")
    train_ids = set(cache["start_ids"][cache["split"] == "train"])
    val_ids = set(cache["start_ids"][cache["split"] == "val"])
    if train_ids & val_ids:
        raise ValueError(f"train/validation start ID overlap: {sorted(train_ids & val_ids)}")
    train_costs = cache["true_costs"][cache["split"] == "train"]
    if not np.any(np.ptp(train_costs, axis=1) > 0):
        raise ValueError("no observed within-start preference pairs; collect differing outcomes first")
    cache["artifact_scope"] = scope
    return cache


def _tensors(cache, device):
    return {key: torch.as_tensor(cache[key], device=device,
                                dtype=torch.int64 if key == "candidate_ids" else torch.float32)
            for key in ("terminal", "goals", "native_costs", "candidate_ids", "true_costs")}


def selection_metrics(scores, costs, ids):
    chosen = np.array([np.lexsort((i, s))[0] for s, i in zip(scores, ids)])
    selected = costs[np.arange(len(chosen)), chosen]
    return {"mean_selected_measured_cost": float(selected.mean()),
            "mean_within_pool_regret": float((selected - costs.min(axis=1)).mean()),
            "decisions": len(chosen)}


def fit_alignment(cache, *, epochs=200, seed=0, learning_rate=1e-3, batch_size=8, device="cpu"):
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid training settings")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, _, g, d = cache["terminal"].shape
    model = RelationalAlignment(d, g).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    tensors = _tensors(cache, device)
    train, val = np.flatnonzero(cache["split"] == "train"), np.flatnonzero(cache["split"] == "val")
    history = []
    for epoch in range(epochs):
        model.train()
        shuffled, losses = rng.permutation(train), []
        for begin in range(0, len(shuffled), batch_size):
            idx = torch.as_tensor(shuffled[begin:begin + batch_size], device=device)
            y = tensors["true_costs"][idx]
            preference = y[:, :, None] < y[:, None, :]
            if not preference.any():
                continue
            out = model(tensors["terminal"][idx], tensors["goals"][idx], tensors["candidate_ids"][idx],
                        native_costs=tensors["native_costs"][idx])
            scores = out["scores"]
            loss = F.softplus((scores[:, :, None] - scores[:, None, :])[preference] / 0.1).mean()
            loss = loss + 1e-3 * out["correction"].square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            vi = torch.as_tensor(val, device=device)
            out = model(tensors["terminal"][vi], tensors["goals"][vi], tensors["candidate_ids"][vi],
                        native_costs=tensors["native_costs"][vi])
            values = dict(epoch=epoch + 1, train_pair_loss=float(np.mean(losses)))
            for name, field in (("native", "base_scores"), ("aligned", "scores")):
                values[name] = selection_metrics(out[field].cpu().numpy(),
                                                 cache["true_costs"][val], cache["candidate_ids"][val])
            history.append(values)
    model.cpu().eval()
    receipt = {"artifact_scope": cache["artifact_scope"], "source": cache["source"],
               "feature_spec": cache["feature_spec"], "source_model": cache["source_model"],
               "seed": seed, "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
               "train_start_ids": sorted({str(s) for s in cache["start_ids"][train]}),
               "val_start_ids": sorted({str(s) for s in cache["start_ids"][val]}), "history": history,
               "checkpoint_selection": "last epoch; validation reported, not formal evaluation",
               "label_definition": "measured true_costs, lower is better",
               "evidence_check": "nonempty references checked, referenced rollout content not independently audited",
               "hardware_execution": False}
    checkpoint = {"format": "djepa_robot_relational_v1", "config": model.config,
                  "state_dict": model.state_dict(), "metadata": receipt}
    return checkpoint, receipt


def load_alignment(path, *, allow_synthetic=False, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "djepa_robot_relational_v1":
        raise ValueError("wrong alignment checkpoint format")
    metadata = checkpoint["metadata"]
    if metadata["artifact_scope"] == SYNTHETIC_SCOPE and not allow_synthetic:
        raise ValueError("synthetic test checkpoint requires explicit opt-in; never a robot model")
    model = RelationalAlignment(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), metadata
