# src/models/neural_web.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train_neural_network(x_train, y_train):
    """
    Treina uma Rede Neural MLP para classificação de diabetes.

    Args:
        x_train: Dados de entrada de treinamento
        y_train: Rótulos de treinamento

    Returns:
        model: Modelo treinado contendo o scaler e o classificador
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    mlp.fit(x_train_scaled, y_train)

    class Model:
        def __init__(self, scaler, classifier):
            self.scaler = scaler
            self.classifier = classifier

        def predict(self, x):
            x_scaled = self.scaler.transform(x)
            return self.classifier.predict(x_scaled)

    return Model(scaler, mlp)
