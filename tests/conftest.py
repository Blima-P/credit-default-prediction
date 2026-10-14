import numpy as np
import pandas as pd
import pytest

from explore_data import DATA_FILE, TARGET


@pytest.fixture
def fake_data() -> pd.DataFrame:
    """Dados pequenos e controlados: 100 clientes, 20% inadimplentes."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "AGE": rng.integers(21, 70, n),
            TARGET: [1] * 20 + [0] * 80,
        }
    )


@pytest.fixture(scope="session")
def real_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        pytest.skip("Dataset real ausente; rode src\\explore_data.py antes.")
    from explore_data import load_data

    return load_data()
