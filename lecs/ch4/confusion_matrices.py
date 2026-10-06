# DATA 419 fall 2026
# Some basic confusion-matrix-oriented stats.
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ground_truth = np.array([1,0,0,0,1,1,1,0])
classifier1 = np.array([1,1,1,0,1,1,0,1])
classifier2 = np.array([0,0,0,0,1,0,0,0])

print('Classifier 1:')
print(confusion_matrix(ground_truth, classifier1))
print(f"Accuracy: {accuracy_score(ground_truth,classifier1):.3f}")
print(f"Precision: {precision_score(ground_truth,classifier1):.3f}")
print(f"Recall: {recall_score(ground_truth,classifier1):.3f}")
print(f"f1: {f1_score(ground_truth,classifier1):.3f}")

print('Classifier 2:')
print(confusion_matrix(ground_truth, classifier2))
print(f"Accuracy: {accuracy_score(ground_truth,classifier2):.3f}")
print(f"Precision: {precision_score(ground_truth,classifier2):.3f}")
print(f"Recall: {recall_score(ground_truth,classifier2):.3f}")
print(f"f1: {f1_score(ground_truth,classifier2):.3f}")
