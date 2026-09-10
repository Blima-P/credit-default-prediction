from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "default of credit card clients.xls"
)
TARGET = "default payment next month"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        print("Baixando o dataset publico da UCI...")
        with urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            spreadsheet = archive.read(DATA_FILE.name)
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_bytes(spreadsheet)

    # A primeira linha da planilha agrupa colunas; a segunda tem os nomes.
    data = pd.read_excel(DATA_FILE, header=1)
    if TARGET not in data.columns or "ID" not in data.columns:
        raise ValueError("A planilha nao tem as colunas esperadas.")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("O alvo deve conter apenas 0 ou 1, sem valores ausentes.")
    return data


def main() -> None:
    data = load_data()
    features = data.drop(columns=["ID", TARGET])
    target = data[TARGET]

    print(f"\nClientes: {len(data):,}")
    print(f"Variaveis de entrada (sem ID e alvo): {features.shape[1]}")
    print(f"Valores ausentes: {data.isna().sum().sum()}")
    print(f"Proporcao de inadimplentes: {target.mean():.2%}")
    print("\nPrimeiros cinco registros:")
    print(data.head().to_string(index=False))
    print("\nPerguntas para sua primeira analise:")
    print("1. O que estamos tentando prever?")
    print("2. Por que ID nao deve ser uma variavel de entrada?")
    print("3. Qual seria a acuracia de prever sempre 'nao inadimplente'?")


if __name__ == "__main__":
    main()
