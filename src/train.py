import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import load_data
from src.processing import get_preprocessing_pipeline


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\n Training {name}...")

    preprocessor = get_preprocessing_pipeline()

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f" {name} Accuracy: {acc:.4f}")

    return clf, acc


def main():
    print(" Loading data...")
    X_train, y_train, X_test, y_test = load_data()


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    lr_model = LogisticRegression(max_iter=300)


    rf_clf, rf_acc = train_and_evaluate(
        rf_model, "Random Forest", X_train, y_train, X_test, y_test
    )

    lr_clf, lr_acc = train_and_evaluate(
        lr_model, "Logistic Regression", X_train, y_train, X_test, y_test
    )


    if rf_acc > lr_acc:
        best_model = rf_clf
        best_name = "Random Forest"
        best_acc = rf_acc
    else:
        best_model = lr_clf
        best_name = "Logistic Regression"
        best_acc = lr_acc

    print(f"\n Best Model: {best_name} ({best_acc:.4f})")

    joblib.dump(best_model, "best_model.joblib")
    print(" Best model saved as best_model.joblib")


if __name__ == "__main__":
    main()