import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

import data_process as data

from models import (
    train_bayesian_network,
    train_mlp_model
)

def main(x, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "Bayesian": train_bayesian_network,
        "mlp" : train_mlp_model
    }

    results = {name: [] for name in models}

    for fold, (train_val_index, test_index) in enumerate(kf.split(x), 1):
        print(f"\n=== Fold {fold} ===")

        x_test, y_test = x[test_index], y[test_index]
        x_train_val, y_train_val = x[train_val_index], y[train_val_index]

        train_size = int(0.8235 * len(x_train_val))
        x_train = x_train_val[:train_size]
        y_train = y_train_val[:train_size]
        x_val = x_train_val[train_size:]
        y_val = y_train_val[train_size:]

        for name, train_func in models.items():
            print(f"\n>>> Treinando modelo: {name}")
            model = train_func(x_train, y_train, x_val, y_val)

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
    x = data.data_train
    y = data.data_target
    main(x, y)
