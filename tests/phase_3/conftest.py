"""Shared fixtures for Phase 3 (decoding, quantization, serving)."""

import pytest
import torch

from tests.adapters import get_mini_llama
from tests.phase_3.common import SMALL_CONFIG


@pytest.fixture
def small_llama():
    """A freshly initialised (untrained) Mini-Llama in eval mode."""
    torch.manual_seed(0)
    return get_mini_llama(**SMALL_CONFIG).eval()
