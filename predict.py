
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score"
]


def get_number(message, minimum, maximum):
    while True:
        try:
            value = float(input(message))

            if minimum <= value <= maximum:
                return value

            print(f"Enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run python main.py first.")
        return

    model = joblib.load(MODEL_PATH)

    print("Student Pass/Fail Prediction")
    print("Educational demonstration only.\n")

    values = {
        "Study_Hours": get_number("Study hours (0-24): ", 0, 24),
        "Attendance": get_number("Attendance (0-100): ", 0, 100),
        "Previous_Marks": get_number("Previous marks (0-100): ", 0, 100),
        "Assignment_Score": get_number("Assignment score (0-100): ", 0, 100)
    }

    sample = pd.DataFrame([values], columns=FEATURES)

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("\nPredicted Result:", prediction)
    print("\nClass probabilities:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{label}: {probability:.2%}")

    print("\nProbabilities are not guarantees of actual results.")


if __name__ == "__main__":
    main()
