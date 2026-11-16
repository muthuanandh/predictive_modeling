
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# Project folders
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "visualizations"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

DATA_PATH = ROOT / "student_performance.csv"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score"
]


# 1. Create synthetic dataset
def create_dataset():
    np.random.seed(42)
    rows = 1500

    study = np.round(np.random.uniform(0, 8, rows), 1)
    attendance = np.round(np.random.uniform(40, 100, rows), 1)
    previous = np.round(np.random.uniform(20, 100, rows), 1)
    assignment = np.round(np.random.uniform(20, 100, rows), 1)

    final_score = (
        0.24 * (study / 8 * 100)
        + 0.24 * attendance
        + 0.27 * previous
        + 0.25 * assignment
        + np.random.normal(0, 7, rows)
    )

    final_marks = np.clip(
        np.round(final_score), 0, 100
    ).astype(int)

    result = np.where(
        final_marks >= 50, "Pass", "Fail"
    )

    df = pd.DataFrame({
        "Student_ID": [
            f"S{i:04d}" for i in range(1, rows + 1)
        ],
        "Study_Hours": study,
        "Attendance": attendance,
        "Previous_Marks": previous,
        "Assignment_Score": assignment,
        "Final_Marks": final_marks,
        "Result": result
    })

    df.to_csv(DATA_PATH, index=False)
    print("Dataset created successfully!")


# 2. Load and inspect dataset
def load_data():
    if not DATA_PATH.exists():
        create_dataset()

    df = pd.read_csv(DATA_PATH)

    if df.empty or df.isnull().any().any():
        raise ValueError("Dataset is empty or has missing values.")

    required = FEATURES + ["Final_Marks", "Result"]

    if not all(col in df.columns for col in required):
        raise ValueError("Required dataset columns are missing.")

    if not df["Result"].isin(["Pass", "Fail"]).all():
        raise ValueError("Result must be Pass or Fail.")

    if not (df["Result"] == np.where(
        df["Final_Marks"] >= 50, "Pass", "Fail"
    )).all():
        raise ValueError("Incorrect Result labels.")

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nClass distribution:")
    print(df["Result"].value_counts())

    return df


# 3. Create visualizations
def create_charts(df, results, models, X_test, y_test, best_name):

    sns.set_theme(style="whitegrid")

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df, x="Result",
        order=["Fail", "Pass"], color="steelblue"
    )
    plt.title("Pass and Fail Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_distribution.png")
    plt.close()

    # Accuracy comparison
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=results, x="Model",
        y="Accuracy", color="cornflowerblue"
    )
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_accuracy.png")
    plt.close()

    # Precision, recall, F1
    metrics = results.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=metrics, x="Model",
        y="Score", hue="Metric"
    )
    plt.ylim(0, 1)
    plt.title("Model Metrics Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_metrics.png")
    plt.close()

    # Confusion matrix
    best_model = models[best_name]
    predictions = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test, predictions,
        labels=["Fail", "Pass"]
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png")
    plt.close()

    # ROC curve comparison
    plt.figure(figsize=(7, 5))

    y_binary = (y_test == "Pass").astype(int)

    for name, model in models.items():
        probabilities = model.predict_proba(X_test)
        pass_index = list(model.classes_).index("Pass")
        probabilities = probabilities[:, pass_index]

        fpr, tpr, _ = roc_curve(
            y_binary, probabilities
        )

        auc = roc_auc_score(y_binary, probabilities)

        plt.plot(
            fpr, tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve.png")
    plt.close()

    # Random Forest feature importance
    rf = models["Random Forest"]
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=importance,
        x="Importance", y="Feature",
        color="seagreen"
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png")
    plt.close()

    # Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.barplot(
        data=results, x="Model",
        y="Accuracy", color="cornflowerblue",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_ylim(0, 1)

    sns.barplot(
        data=metrics, x="Model",
        y="Score", hue="Metric", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Precision / Recall / F1")
    axes[0, 1].set_ylim(0, 1)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Confusion Matrix")

    sns.countplot(
        data=df, x="Result",
        order=["Fail", "Pass"],
        color="orange", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Class Distribution")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_dashboard.png")
    plt.close(fig)


# 4. Train and evaluate models
def main():
    df = load_data()

    # ID and Final_Marks are excluded from model inputs.
    X = df[FEATURES]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )
    }

    records = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)
        pass_index = list(model.classes_).index("Pass")
        pass_probabilities = probabilities[:, pass_index]

        y_binary = (y_test == "Pass").astype(int)

        records[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions,
                pos_label="Pass", zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions,
                pos_label="Pass", zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, predictions,
                pos_label="Pass", zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y_binary, pass_probabilities
            )
        }

        print(f"\n--- {name} ---")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

    results = pd.DataFrame.from_dict(
        records, orient="index"
    ).rename_axis("Model").reset_index()

    results = results.sort_values(
        ["F1_Score", "ROC_AUC"],
        ascending=False
    ).reset_index(drop=True)

    best_name = results.iloc[0]["Model"]

    # Save results and best model
    results.to_csv(ROOT / "model_comparison.csv", index=False)
    joblib.dump(
        models[best_name],
        MODEL_DIR / "best_model.joblib"
    )

    report = (
        "PREDICTIVE MODELING EVALUATION REPORT\n\n"
        + results.to_string(index=False)
        + f"\n\nSelected model: {best_name}\n"
        + "Selection criterion: highest test F1-score; ROC-AUC breaks ties.\n"
        + "Data is synthetic and results are for educational demonstration only.\n"
    )

    (ROOT / "evaluation_report.txt").write_text(
        report, encoding="utf-8"
    )

    create_charts(
        df, results, models, X_test, y_test, best_name
    )

    print("\nMODEL COMPARISON:")
    print(results.to_string(index=False))
    print("\nBest model:", best_name)
    print("Model and reports saved successfully.")
    print("Charts saved in the visualizations folder.")


if __name__ == "__main__":
    main()
