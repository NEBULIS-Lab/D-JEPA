# Task-interface checks

The robotic-manipulation and driving interfaces were checked on 2026-09-17.

- 75 CPU tests passed across `tests/robotics/` and `tests/driving/`.
- Robot tests cover RLDS structure, data splits, action conversion, geometry
  fitting, candidate composition, preservation selection and report interfaces.
- Driving tests cover zero-correction native preservation, permutation
  behavior, bounded relative corrections, risk vetoes, source-log isolation,
  prediction-only input shapes and recipe argument construction.
- Training/calibration/native-export CLI help and YAML dry-run commands passed.
- `examples/driving_label_free.py` passed using synthetic inputs and an
  untrained zero-correction head, without simulator execution.
- Website resource/link checks passed after adding the application guides.

No experiment training, simulator rollout, model-file audit or numerical-result
re-evaluation was performed for this code integration. Third-party environment
dependencies remain external and are documented in the corresponding guides.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/robotics tests/driving -q
python examples/driving_label_free.py
python scripts/run_task.py --config configs/driving/relational.yaml --dry-run
```
