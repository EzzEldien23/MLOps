import joblib
import hydra
from omegaconf import DictConfig

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.processing import get_preprocessing_pipeline
from src.utils import load_data


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\n Training {name}...")

    preprocessor = get_preprocessing_pipeline()

    clf = Pipeline([("preprocessor", preprocessor), ("model", model)])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f" {name} Accuracy: {acc:.4f}")

    return clf, acc


@hydra.main(config_path="../config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    print(" Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    if cfg.model.name == "random_forest":
        model = RandomForestClassifier(**cfg.model.random_forest)
        model_name = "Random Forest"

    elif cfg.model.name == "logistic_regression":
        model = LogisticRegression(**cfg.model.logistic_regression)
        model_name = "Logistic Regression"

    else:
        raise ValueError("Unknown model!")

    clf, acc = train_and_evaluate(
        model, model_name, X_train, y_train, X_test, y_test
    )

    print(f"\n Final Model: {model_name} ({acc:.4f})")

    joblib.dump(clf, 'models/model.joblib')
    print(f" Model saved at {'models/model.joblib'}")


if __name__ == "__main__":
    main()