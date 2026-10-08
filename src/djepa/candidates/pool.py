"""Action pools with persistent IDs and an explicit reference policy."""
from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class CandidatePool:
    ids: np.ndarray
    actions: np.ndarray
    types: np.ndarray
    profile: str
    seed: int | None = None
    reference_id: int | None = 0

    def __post_init__(self):
        if (self.ids.ndim != 1 or self.ids.dtype != np.int64 or len(self.ids) == 0
                or len(np.unique(self.ids)) != len(self.ids) or np.any(self.ids < 0)):
            raise ValueError('candidate IDs must be distinct nonnegative int64 values')
        if (self.actions.ndim != 3 or self.actions.shape[0] != len(self.ids)
                or min(self.actions.shape[1:]) < 1 or self.actions.dtype.kind != 'f'
                or not np.isfinite(self.actions).all()):
            raise ValueError('candidate actions must be finite floating-point [K,H,D]')
        if self.types.shape != self.ids.shape or self.types.dtype.kind not in 'US':
            raise ValueError('candidate types must be a matching string array')
        if self.reference_id is not None and self.reference_id not in self.ids:
            raise ValueError('reference ID is absent from the pool')

    def subset(self, positions, *, reference_id=None):
        return CandidatePool(self.ids[positions].copy(), self.actions[positions].copy(),
                             self.types[positions].copy(), self.profile, self.seed, reference_id)

    def without_reference(self):
        """Remove only the designated reference; Granular's replay anchor stays."""
        if self.reference_id is None:
            return self
        return self.subset(np.flatnonzero(self.ids != self.reference_id))

    def action_hashes(self):
        """Canonical float32 action hashes, matching the native PushT interface."""
        return np.asarray([hashlib.sha256(np.ascontiguousarray(a, dtype='<f4').tobytes()).hexdigest()
                           for a in self.actions], dtype='S64')
