from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from explore_data import TARGET, load_data


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Estratificar mantem a mesma proporcao de inadimplentes nos dois conjuntos.
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    if set(train["ID"]) & set(test["ID"]):
        raise ValueError("Ha clientes presentes no treino e no teste.")
    return train, test


def main() -> None:
    data = load_data()
    train, test = split_data(data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print(f"{'Conjunto':<10}{'Clientes':>10}{'Inadimplentes':>16}")
    for name, subset in [("Total", data), ("Treino", train), ("Teste", test)]:
        print(f"{name:<10}{len(subset):>10,}{subset[TARGET].mean():>16.2%}")
    print(f"\nArquivos salvos em {PROCESSED_DIR}")
    print("Use apenas o treino para explorar e ajustar modelos.")
    print("O teste fica reservado para a avaliacao final.")


if __name__ == "__main__":
    main()
