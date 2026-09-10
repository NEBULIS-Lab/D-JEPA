"""Descriptive command tracking, with matched valid timestamp windows."""

import numpy as np

from .data import valid_window_starts


def tracking_scan(states, commands, timestamps, *, max_lag=5, min_dt=.08, max_dt=.15):
    states, commands = np.asarray(states), np.asarray(commands)
    if states.ndim != 2 or states.shape != commands.shape or len(states) != len(timestamps):
        raise ValueError("state/command/time dimensions must match")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("tracking arrays must be finite")
    starts = valid_window_starts(timestamps, max_lag, min_dt, max_dt)
    if not len(starts):
        return {"common_window_count": 0, "joint_mae_rad_by_lag": [],
                "best_descriptive_lag_steps": None, "identified_physical_delay": False}
    errors = [float(np.abs(commands[starts] - states[starts + lag]).mean()) for lag in range(max_lag + 1)]
    elapsed = [float(np.median((timestamps[starts + lag] - timestamps[starts]) * 1e-9))
               for lag in range(max_lag + 1)]
    return {"common_window_count": len(starts), "joint_mae_rad_by_lag": errors,
            "median_elapsed_s_by_lag": elapsed, "best_descriptive_lag_steps": int(np.argmin(errors)),
            "identified_physical_delay": False,
            "interpretation": "matched-window descriptive tracking; not a causal latency estimator"}
