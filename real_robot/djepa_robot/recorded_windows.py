"""Match cached visual identities to consecutive, timing-valid physical frames."""

import numpy as np

from .data import valid_window_starts


def cached_transition_pairs(frame_indices, timestamps, *, min_dt, max_dt):
    ids = np.asarray(frame_indices)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(np.unique(ids)):
        raise ValueError("unique integer cached frame IDs required")
    if np.any(ids < 0) or np.any(ids >= len(timestamps)):
        raise ValueError("cached frame IDs outside original episode")
    valid = set(valid_window_starts(timestamps, 1, min_dt, max_dt).tolist())
    positions = {int(frame): index for index, frame in enumerate(ids)}
    return [(positions[frame], positions[frame+1], frame) for frame in sorted(positions)
            if frame in valid and frame+1 in positions]
