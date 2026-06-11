import joblib
import hydra
import optuna
from omegaconf import DictConfig
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    log_loss,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.processing import get_preprocessing_pipeline
from src.utils import (
    load_data,
    get_project_root,
    get_git_commit,
    get_dataset_version,
    setup_mlflow_tracking,
)
import mlflow
import mlflow.sklearn


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\n Training {name}...")

    preprocessor = get_preprocessing_pipeline()

    clf = Pipeline([("preprocessor", preprocessor), ("model", model)])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f" {name} Accuracy: {acc:.4f}")

    return clf, acc


def tune_hyperparameters(model_name: str, X_train, y_train, n_trials: int = 20):
    print(f"\n Tuning hyperparameters for {model_name} with Optuna ({n_trials} trials)...")

    preprocessor = get_preprocessing_pipeline()

    def objective(trial):
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
            }
            model = RandomForestClassifier(**params)
        elif model_name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "random_state": 42,
            }
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")

        clf = Pipeline([("preprocessor", preprocessor), ("model", model)])
        
        # Track each trial as a nested MLflow run
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_param("model_architecture", str(clf))
                
                # Compute CV score
                scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
                mean_cv_acc = scores.mean()
                mlflow.log_metric("cv_accuracy", mean_cv_acc)
                
                # Compute and log training/validation loss on a single split
                try:
                    from sklearn.model_selection import train_test_split
                    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    clf.fit(X_tr, y_tr)
                    tr_loss = log_loss(y_tr, clf.predict_proba(X_tr))
                    val_loss = log_loss(y_val, clf.predict_proba(X_val))
                    mlflow.log_metric("training_loss_epoch", tr_loss, step=1)
                    mlflow.log_metric("validation_loss_epoch", val_loss, step=1)
                except Exception as loss_err:
                    print(f"⚠️ Error computing loss for trial: {loss_err}")
                
                mlflow.log_metric("learning_rate", 0.0) # default/Not applicable
                
                return mean_cv_acc
        except Exception as e:
            print(f"⚠️ Error tracking trial in MLflow: {e}")
            # Fallback to local evaluation without MLflow
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
            return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f" Best trial score: {study.best_value:.4f}")
    print(" Best parameters:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    return study.best_params


@hydra.main(config_path="../config", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    print(" Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    model_name = cfg.model.name

    # Initialize MLflow tracking and DagsHub credentials
    setup_mlflow_tracking()

    experiment_name = "Titanic-Classification"
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"[WARNING] Failed to set MLflow experiment: {e}. Falling back to local tracking.")
        root = get_project_root()
        local_uri = f"file:///{os.path.join(root, 'mlruns').replace(os.sep, '/')}"
        mlflow.set_tracking_uri(local_uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as local_err:
            print(f"[WARNING] Local MLflow experiment setup failed: {local_err}")

    # Start active MLflow run
    active_run = None
    try:
        display_name_init = "Random Forest" if model_name == "random_forest" else "Logistic Regression"
        if getattr(cfg, "tune", False):
            display_name_init += " (Tuned)"
        run_name = f"{display_name_init}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        active_run = mlflow.start_run(run_name=run_name)
        print(f"[START] Started MLflow run: {run_name}")
    except Exception as e:
        print(f"[WARNING] Could not start MLflow run: {e}")

    # Determine hyperparameters and train model
    best_params = None
    if getattr(cfg, "tune", False):
        n_trials = getattr(cfg, "n_trials", 20)
        best_params = tune_hyperparameters(model_name, X_train, y_train, n_trials=n_trials)
        
        if model_name == "random_forest":
            model = RandomForestClassifier(**best_params)
            display_name = "Random Forest (Tuned)"
        elif model_name == "logistic_regression":
            model = LogisticRegression(**best_params)
            display_name = "Logistic Regression (Tuned)"
        else:
            raise ValueError("Unknown model!")
        params_to_log = best_params
    else:
        if model_name == "random_forest":
            model = RandomForestClassifier(**cfg.model.random_forest)
            display_name = "Random Forest"
            params_to_log = dict(cfg.model.random_forest)
        elif model_name == "logistic_regression":
            model = LogisticRegression(**cfg.model.logistic_regression)
            display_name = "Logistic Regression"
            params_to_log = dict(cfg.model.logistic_regression)
        else:
            raise ValueError("Unknown model!")
        params_to_log["tune"] = False

    clf, acc = train_and_evaluate(
        model, display_name, X_train, y_train, X_test, y_test
    )

    print(f"\n Final Model: {display_name} ({acc:.4f})")

    # Use configured model path
    model_path = cfg.output.model_path if "output" in cfg and "model_path" in cfg.output else "models/model.joblib"
    project_root = get_project_root()
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f" Model saved at {model_path}")

    # Log parameters, metrics, and artifacts to MLflow if run is active
    if active_run:
        try:
            # Retrieve metadata
            git_commit = get_git_commit()
            dataset_version = get_dataset_version()
            training_date = datetime.date.today().isoformat()

            # Set tags
            tags = {
                "project_name": "MLOps",
                "model_name": display_name,
                "dataset_version": dataset_version,
                "git_commit": git_commit,
                "training_date": training_date,
            }
            mlflow.set_tags(tags)

            # Log parameters
            mlflow.log_params(params_to_log)
            mlflow.log_param("model_architecture", str(clf))

            # Calculate and log train/val loss
            y_train_pred_prob = clf.predict_proba(X_train)
            y_test_pred_prob = clf.predict_proba(X_test)
            train_loss = log_loss(y_train, y_train_pred_prob)
            val_loss = log_loss(y_test, y_test_pred_prob)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("training_loss_epoch", train_loss, step=1)
            mlflow.log_metric("validation_loss_epoch", val_loss, step=1)
            mlflow.log_metric("learning_rate", 0.0) # default/Not applicable

            # Log model checkpoint artifact
            mlflow.log_artifact(model_path, artifact_path="checkpoints")

            # Log sklearn model artifact
            mlflow.sklearn.log_model(clf, artifact_path="model")

            # Generate and log Confusion Matrix
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {display_name}")
            fig_dir = os.path.join(project_root, "reports", "figures")
            os.makedirs(fig_dir, exist_ok=True)
            cm_path = os.path.join(fig_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # Generate and log ROC Curve
            RocCurveDisplay.from_estimator(clf, X_test, y_test)
            plt.title(f"ROC Curve - {display_name}")
            roc_path = os.path.join(fig_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # Generate and log Classification Report
            report = classification_report(y_test, y_pred)
            report_path = os.path.join(project_root, "reports", "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # Generate and log Prediction Samples CSV
            samples_df = X_test.copy()
            samples_df["true_label"] = y_test
            samples_df["predicted_label"] = y_pred
            samples_path = os.path.join(fig_dir, "prediction_samples.csv")
            samples_df.head(20).to_csv(samples_path, index=False)
            mlflow.log_artifact(samples_path, artifact_path="samples")

            print("[OK] Successfully logged parameters, metrics, and artifacts to MLflow.")
        except Exception as e:
            print(f"[WARNING] Error logging to MLflow: {e}")
        finally:
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"[WARNING] Error ending MLflow run: {e}")

    # Best Model Selection and Registry
    try:
        print("\n Retrieving all runs to select the best model...")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs_df.empty:
                finished_runs = runs_df[runs_df["status"] == "FINISHED"]
                primary_metric = "metrics.accuracy"
                secondary_metric = "metrics.validation_loss_epoch"

                if primary_metric in finished_runs.columns:
                    valid_runs = finished_runs.dropna(subset=[primary_metric])
                    sort_cols = [primary_metric]
                    ascending_flags = [False]

                    if secondary_metric in valid_runs.columns:
                        sort_cols.append(secondary_metric)
                        ascending_flags.append(True)

                    valid_runs = valid_runs.sort_values(by=sort_cols, ascending=ascending_flags)
                    best_run_row = valid_runs.iloc[0]
                    best_run_id = best_run_row["run_id"]
                    best_accuracy = best_run_row[primary_metric]

                    print(f"[BEST] Best run selected: ID={best_run_id} with Accuracy={best_accuracy:.4f}")
                    
                    registered_model_name = "TitanicClassifier"
                    model_uri = f"runs:/{best_run_id}/model"
                    print(f" Registering model from run {best_run_id} as '{registered_model_name}'...")
                    
                    model_version = mlflow.register_model(model_uri, registered_model_name)
                    print(f"[OK] Registered model version: {model_version.version}")
                else:
                    print("[WARNING] No runs with metric 'accuracy' found. Skipping model registry.")
            else:
                print("[WARNING] No runs found in experiment. Skipping model registry.")
    except Exception as e:
        print(f"[ERROR] Error during model retrieval/registration: {e}")


if __name__ == "__main__":
    main()