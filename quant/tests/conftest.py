import copy

import pytest

from quant.engine.config import load_config
from quant.engine.data import synthetic


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


@pytest.fixture
def df():
    return synthetic(n_bars=3000, seed=7)
