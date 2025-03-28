from .bayesian_network import train_bayesian_network
from .neural_web import train_neural_network
from .random_forest import train_random_forest
from .knn import train_knn_classifier
from .svm import train_svm_classifier

__all__ = [
    "train_bayesian_network",
    "train_neural_network",
    "train_random_forest",
    "train_knn_classifier",
    "train_svm_classifier"
]