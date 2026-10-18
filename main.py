from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from classifiers.gaussian_nb import create_gaussian_nb
from classifiers.knn import create_knn
from load import load_data
from preprocess import preprocess_data
from validation import evaluate_classifier


def main():
    raw = load_data("data/weatherAUS.csv")
    df = preprocess_data(raw)
    X, y = df.drop(columns="RainTomorrow"), df["RainTomorrow"]

    print(f"Použito {len(df)} z {len(raw)} řádků, {X.shape[1]} features.")
    print(f"Třídy: No={(y == 0).sum()}, Yes={(y == 1).sum()}")
    print(f"Accuracy při předpovědi vždy No: {(y == 0).mean():.4f}")

    output = Path("results")
    output.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for axis, (name, classifier) in zip(axes, [
        ("GaussianNB", create_gaussian_nb()), ("kNN", create_knn()),
    ]):
        result = evaluate_classifier(classifier, X, y, name)
        metrics = result["fold_metrics"]
        metrics.to_csv(output / f"{name}_fold_metrics.csv")

        print(f"\n{name} – průměr a směrodatná odchylka přes 10 foldů:")
        print(metrics.agg(["mean", "std"]).round(4).to_string())
        print(classification_report(
            y, result["predictions"], labels=[0, 1], target_names=["No", "Yes"],
            digits=4, zero_division=0,
        ))
        print("Confusion matrix (řádky = skutečnost, sloupce = predikce; No, Yes):")
        print(result["confusion_matrix"])

        ConfusionMatrixDisplay(
            result["confusion_matrix"], display_labels=["No", "Yes"],
        ).plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
        axis.set_title(name)
        axis.set_xlabel("Predikovaná třída")
        axis.set_ylabel("Skutečná třída")

    figure.tight_layout()
    figure.savefig(output / "confusion_matrices.png", dpi=160)

    print(f"\nVýsledky uloženy do: {output}")
    plt.show()


if __name__ == "__main__":
    main()
