from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_knn_classifier(X_train, y_train, X_val=None, y_val=None, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print("Validação - Acurácia:", acc)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))
    return model
