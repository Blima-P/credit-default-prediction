import pandas as pd
import pytest

from explore_data import TARGET
from split_data import TEST_SIZE, split_data


def test_tamanhos_corretos(fake_data):
    train, test = split_data(fake_data)
    assert len(test) == int(len(fake_data) * TEST_SIZE)
    assert len(train) + len(test) == len(fake_data)


def test_nenhum_cliente_em_ambos(fake_data):
    train, test = split_data(fake_data)
    assert set(train["ID"]).isdisjoint(test["ID"])


def test_nenhum_cliente_perdido(fake_data):
    train, test = split_data(fake_data)
    assert set(train["ID"]) | set(test["ID"]) == set(fake_data["ID"])


def test_estratificacao_preserva_proporcao(fake_data):
    train, test = split_data(fake_data)
    assert train[TARGET].mean() == pytest.approx(0.20)
    assert test[TARGET].mean() == pytest.approx(0.20)


def test_resultado_reprodutivel(fake_data):
    train_a, test_a = split_data(fake_data)
    train_b, test_b = split_data(fake_data)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


@pytest.mark.real_data
def test_split_real(real_data):
    train, test = split_data(real_data)
    assert (len(train), len(test)) == (24_000, 6_000)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.001
