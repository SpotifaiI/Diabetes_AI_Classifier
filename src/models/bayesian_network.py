from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def train_bayesian_network(X_train, y_train, X_val=None, y_val=None):
    model = GaussianNB() # Rede bayesiana simples
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))

    return model