import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_random_forest(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(x_train_scaled, y_train)

    class Model:
        def __init__(self, scaler, classifier):
            self.scaler = scaler
            self.classifier = classifier

        def predict(self, x):
            x_scaled = self.scaler.transform(x)
            return self.classifier.predict(x_scaled)

    return Model(scaler, rf)
