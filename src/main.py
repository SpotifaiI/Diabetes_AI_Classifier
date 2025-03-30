import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import uuid

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import data_process as data
from models import (
    train_bayesian_network,
    train_neural_network,
    train_random_forest,
    train_knn_classifier,
    train_svm_classifier
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
        "Neural Network": train_neural_network,
        "Random Forest": train_random_forest,
        "KNN": train_knn_classifier,
        "SVM": train_svm_classifier
    }
    models_acc_list = {model_name: [] for model_name in models}

    roc_curves = {}

    for model_name, train_model in models.items():
        logging.info("=== Training and rating model : %s ===", model_name)
        y_true_all = []
        y_score_all = []

        for _, row in df_splits.iterrows():
            fold = row["fold"]
            train_indices = eval(row["train_indices"])
            test_indices = eval(row["test_indices"])

            x_train_val = x.iloc[train_indices]
            y_train_val = y.iloc[train_indices]
            x_test = x.iloc[test_indices]
            y_test = y.iloc[test_indices]

            model = train_model(x_train_val, y_train_val)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            models_acc_list[model_name].append(acc)

            y_true_all.extend(y_test)
            y_score_all.extend(y_pred)

            logging.info(f"Fold {fold} - Accuracy: {acc:.4f}")
            logging.info(classification_report(y_test, y_pred))
        
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        roc_auc = auc(fpr, tpr)
        roc_curves[model_name] = (fpr, tpr, roc_auc)
    
    logging.info("=== Accuracy average results ===")
    for model_name, accs in models_acc_list.items():
        logging.info(f"{model_name}: Average = {np.mean(accs):.4f}, Standard Variation = {np.std(accs):.4f}")

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

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo (FPR)')
    plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
    plt.title('Curva ROC Comparativa')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x, y = data.read_data(891)
    task_id = uuid.uuid4()
    main(x, y, task_id)
