import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_estimator(model, X_val, y_val)
plt.title("Krzywa ROC modelu sejsmicznego")
plt.savefig("roc_curve.png")

PrecisionRecallDisplay.from_estimator(model, X_val, y_val)
plt.title("Krzywa Precision-Recall modelu")
plt.savefig("pr_curve.png")
