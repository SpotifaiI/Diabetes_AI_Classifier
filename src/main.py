import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import uuid
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from sklearn.metrics import accuracy_score, classification_report

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
        logging.info(f"{model_name}: Average = {np.mean(accs):.4f}, Standard Variation = {np.std(accs):.4f}")

    models_name = list(models_acc_list.keys())
    average_acc_per_model = [np.mean(models_acc_list[m]) for m in models_name]
    os.remove(filename)

    plt.figure(figsize=(10, 6))
    for model_name, train_model in models.items():
        tpr_list = []
        fpr_list = []
        mean_auc = 0
        
        for _, row in df_splits.iterrows():
            fold = row["fold"]
            train_indices = eval(row["train_indices"])
            test_indices = eval(row["test_indices"])

            x_train_val = x.iloc[train_indices]
            y_train_val = y.iloc[train_indices]
            x_test = x.iloc[test_indices]
            y_test = y.iloc[test_indices]

            model = train_model(x_train_val, y_train_val)
            
            try:
                y_pred_proba = model.predict_proba(x_test)[:, 1]
            except (AttributeError, NotImplementedError):
                try:
                    y_pred_proba = model.decision_function(x_test)
                except (AttributeError, NotImplementedError):
                    y_pred_proba = model.predict(x_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            mean_auc += roc_auc
            
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            tpr_list.append(mean_tpr)
            fpr_list.append(mean_fpr)

        # Calcula a média das curvas ROC
        mean_tpr = np.mean(tpr_list, axis=0)
        mean_fpr = fpr_list[0]
        mean_auc = mean_auc / len(df_splits)
        
        plt.plot(
            mean_fpr, 
            mean_tpr,
            label=f'{model_name} (AUC = {mean_auc:.2f})',
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC Média para Cada Modelo')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    x, y = data.read_data(891)
    task_id = uuid.uuid4()
    main(x, y, task_id)
