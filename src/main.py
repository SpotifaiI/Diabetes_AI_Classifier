import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import data_process as data

from models import (
    train_bayesian_network
)

def main(x, y):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        "Bayesian": train_bayesian_network,
    }

    results = {name: [] for name in models}

    for fold, (train_val_index, test_index) in enumerate(kf.split(x), 1):
        print(f"\n=== Fold {fold} ===")

        x_train_val = x.iloc[train_val_index]
        y_train_val = y.iloc[train_val_index]
        x_test = x.iloc[test_index]
        y_test = y.iloc[test_index]

        x_train = x_train_val
        y_train = y_train_val
        print(y_train.value_counts(normalize=True))

        for name, train_func in models.items():
            print(f"\n>>> Treinando modelo: {name}")
            model = train_func(x_train, y_train)

            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[{name}] Teste - Acurácia:", acc)
            print(classification_report(y_test, y_pred))
            results[name].append(acc)

    print("\n=== Resultados Médios ===")
    for name, accs in results.items():
        print(f"{name}: Média = {np.mean(accs):.4f}, Desvio Padrão = {np.std(accs):.4f}")

    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    avg_accuracies = [np.mean(acc) for acc in accuracies]
    plt.bar(model_names, avg_accuracies)
    plt.title("Média de Acurácia por Modelo")
    plt.ylabel("Acurácia Média")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    x, y = data.read_data(891)
    main(x, y)
