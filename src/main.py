import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import uuid

from sklearn.metrics import accuracy_score, classification_report

import data_process as data
from models import (
    train_bayesian_network,
    train_neural_network
)

def main(x, y, task_id):
    logging.info(
        "==========Starting process id: %s ================",
        task_id
    )

    filename = 'kfold_splits.csv'
    df_splits = data.gen_kfold_and_save(
        x,
        filename=filename,
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    df_splits = data.load_kfold_df(filename)

    models = {
        "Bayesian": train_bayesian_network,
        "Neural_network" : train_neural_network
        
    }
    models_acc_list = {model_name: [] for model_name in models}

    for model_name, train_model in models.items():
        logging.info("=== Training and rating model : %s ===", model_name)
        
        for _, row in df_splits.iterrows():
            fold = row["fold"]
            train_indices = eval(row["train_indices"])
            test_indices = eval(row["test_indices"])

            x_train_val = x.iloc[train_indices]
            y_train_val = y.iloc[train_indices]
            x_test = x.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # sm = SMOTE(random_state=42)
            # x_train_val, y_train_val = sm.fit_resample(x_train_val, y_train_val)

            model = train_model(x_train_val, y_train_val)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            models_acc_list[model_name].append(acc)

            logging.info(f"Fold {fold} - Accuracy: {acc:.4f}")
            logging.info(classification_report(y_test, y_pred))
    
    logging.info("=== Accuracy average results ===")
    for model_name, accs in models_acc_list.items():
        logging.info(f"{model_name}: Average = {np.mean(accs):.4f}, Standart Variation = {np.std(accs):.4f}")

    models_name = list(models_acc_list.keys())
    average_acc_per_model = [np.mean(models_acc_list[m]) for m in models_name]
    os.remove(filename)

    plt.figure(figsize=(10, 6))
    plt.bar(models_name, average_acc_per_model)
    plt.title("Média de Acurácia por Modelo")
    plt.ylabel("Acurácia Média")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x, y = data.read_data(891)
    task_id = uuid.uuid4()
    main(x, y, task_id)
