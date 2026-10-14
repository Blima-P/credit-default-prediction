import pytest

from explore_data import TARGET, validate_data


def test_dados_validos_passam(fake_data):
    validate_data(fake_data)


def test_falta_coluna_alvo(fake_data):
    with pytest.raises(ValueError, match="colunas esperadas"):
        validate_data(fake_data.drop(columns=[TARGET]))


def test_falta_coluna_id(fake_data):
    with pytest.raises(ValueError, match="colunas esperadas"):
        validate_data(fake_data.drop(columns=["ID"]))


def test_alvo_com_valor_invalido(fake_data):
    fake_data.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="apenas 0 ou 1"):
        validate_data(fake_data)


def test_alvo_com_valor_ausente(fake_data):
    fake_data[TARGET] = fake_data[TARGET].astype("float")
    fake_data.loc[0, TARGET] = None
    with pytest.raises(ValueError, match="apenas 0 ou 1"):
        validate_data(fake_data)


def test_id_repetido(fake_data):
    fake_data.loc[1, "ID"] = fake_data.loc[0, "ID"]
    with pytest.raises(ValueError, match="ID repetido"):
        validate_data(fake_data)


@pytest.mark.real_data
def test_dataset_real_tem_formato_esperado(real_data):
    assert real_data.shape == (30_000, 25)
    assert real_data.isna().sum().sum() == 0
    assert real_data[TARGET].mean() == pytest.approx(0.2212, abs=1e-4)
