"""Label-free inference: python examples/predict_cached.py PROFILE INPUTS.npz."""
import argparse
import numpy as np
from djepa.evaluation.inference import predict

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("profile")
parser.add_argument("inputs")
args = parser.parse_args()
with np.load(args.inputs, allow_pickle=False) as inputs:
    result = predict(args.profile, inputs)
print(result["selected_ids"].tolist())
