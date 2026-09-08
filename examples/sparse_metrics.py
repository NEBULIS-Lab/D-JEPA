"""Unknown outcomes remain unknown; this example performs no model inference."""
import numpy as np
from djepa.evaluate import summarize_selections

report = summarize_selections(
    np.array([0, 1]),
    np.array([[True, False], [False, False]]),
    np.array([[True, False], [True, False]]),
)
print(report)  # 1 success / 1 observed; full-population success is unavailable.
