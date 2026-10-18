import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold


# Funkce pro vyhodnocení klasifikátoru pomocí 10-fold cross validation
def evaluate_classifier(classifier, X, y, name: str) -> dict:
    if len(X) != len(y) or y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("X a y musí mít stejnou délku a cílové třídy musí být 0 a 1.")
    if y.value_counts().min() < 10:
        raise ValueError("Pro 10-fold validaci je potřeba alespoň 10 záznamů každé třídy.")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    predictions = np.empty(len(y), dtype=int)
    folds = []

    for fold, (train, test) in enumerate(cv.split(X, y), start=1):
        model = clone(classifier)
        model.fit(X.iloc[train], y.iloc[train])
        predicted = model.predict(X.iloc[test])
        predictions[test] = predicted
        actual = y.iloc[test]

        folds.append({
            "fold": fold,
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
        })
        
        print(f"{name}: fold {fold}/10 hotov", flush=True)

    return {
        "name": name,
        "fold_metrics": pd.DataFrame(folds).set_index("fold"),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]),
    }
