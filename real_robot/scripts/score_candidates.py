import argparse

import numpy as np
import torch

from djepa_robot.training import load_alignment, load_feature_cache, source_receipt
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description="Score a fixed future-feature pool; labels are not loaded or used")
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    cache = load_feature_cache(args.cache)
    model, metadata = load_alignment(args.checkpoint, allow_synthetic=args.allow_synthetic)
    for key in ("feature_spec", "source_model"):
        if cache[key] != metadata[key]:
            raise ValueError(f"{key} differs between training and inference")
    with torch.inference_mode():
        out = model(torch.tensor(cache["terminal"], dtype=torch.float32),
                    torch.tensor(cache["goals"], dtype=torch.float32),
                    torch.tensor(cache["candidate_ids"], dtype=torch.int64),
                    native_costs=torch.tensor(cache["native_costs"], dtype=torch.float32))
    decisions = []
    for b, ids in enumerate(cache["candidate_ids"]):
        scores, base = out["scores"][b].numpy(), out["base_scores"][b].numpy()
        decisions.append({"decision_index": b, "candidate_ids": ids.tolist(),
                          "native_rank_scores": base.tolist(), "aligned_scores": scores.tolist(),
                          "native_selected_id": int(ids[np.lexsort((ids, base))[0]]),
                          "aligned_selected_id": int(ids[np.lexsort((ids, scores))[0]])})
    output = new_output(args.output)
    write_json(output / "selection.json", {"artifact_scope": metadata["artifact_scope"],
               "hardware_execution": False, "outcomes_used_for_selection": False,
               "source": cache["source"], "checkpoint": source_receipt(args.checkpoint), "decisions": decisions})
    print(output / "selection.json")


if __name__ == "__main__":
    main()
