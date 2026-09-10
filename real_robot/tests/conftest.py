"""Upstream integration is explicit; ordinary tests need no vendor checkout."""
import os
from pathlib import Path

import pytest


@pytest.fixture
def upstream_checkout():
    configured = os.environ.get("DJEPA_VJEPA2_CHECKOUT")
    if not configured:
        pytest.skip("set DJEPA_VJEPA2_CHECKOUT to test the official AC preprocessing")
    path = Path(configured).resolve(strict=True)
    if not (path / "app/vjepa_droid/transforms.py").is_file():
        pytest.fail("DJEPA_VJEPA2_CHECKOUT does not contain the official AC transform")
    return path
