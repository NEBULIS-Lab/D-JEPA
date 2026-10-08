"""CPU-only candidate construction; execution labels are not inputs."""
from .controls import control_candidates, retain_pusht_candidates
from .granular import granular_candidates
from .pool import CandidatePool

__all__ = ['CandidatePool', 'control_candidates', 'retain_pusht_candidates',
           'granular_candidates']
