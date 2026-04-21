import json
import os
import pickle
from typing import Any, Dict

import dvc.api
import dagshub
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dotenv import load_dotenv

from src.logger import ExecutorLogger


def setup_mlflow(tracking_uri: str, logger):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    logger.info("MLFlow Client Defined and tracking URI Setted Successfully.")
    return client


def evaluate(client, cfg: Dict[str, Any], logger) -> None:
    logger.info("loading model")
    test_df = pd.read_parquet(
        os.path.join(
            cfg["evaluate"]["processed_data_path"],
            f"{cfg['evaluate']['file_name']}-test.parquet",
        )
    )
    X_test, y_test = (
        test_df.drop(cfg["evaluate"]["target_column"], axis=1),
        test_df[cfg["evaluate"]["target_column"]],
    )
    with open(
        os.path.join(
            cfg["evaluate"]["model_path"],
            cfg["evaluate"]["model_name"],
            "model_target_translator.pkl",
        ),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])
    version = client.get_latest_versions(name=cfg["evaluate"]["model_name"])[0].version
    logger.info(f"Model Production version: {version}")
    final_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{cfg['evaluate']['model_name']}/{version}"
    )
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": cfg["evaluate"]["model_name"],
        "accuracy": accuracy_score(y_test, final_model.predict(X_test)),
        "precision": precision_score(y_test, final_model.predict(X_test), average="micro"),
        "recall": recall_score(y_test, final_model.predict(X_test), average="micro"),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(
        os.path.join(cfg["evaluate"]["reports_path"], cfg["evaluate"]["model_name"])
    ):
        os.makedirs(
            os.path.join(cfg["evaluate"]["reports_path"], cfg["evaluate"]["model_name"])
        )
    with open(
        os.path.join(
            cfg["evaluate"]["reports_path"],
            cfg["evaluate"]["model_name"],
            "evaluation_report.json",
        ),
        "w",
    ) as js:
        json.dump(evaluation_report, js, indent=4)


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    load_dotenv(".env")
    cfg = dvc.api.params_show()
    logger.info(
        "Paramsters: \n"
        f"{cfg['evaluate']}"
    )
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"), 
        repo_name=cfg["model"]["repo_name"], 
        mlflow=cfg["model"]["use_mlflow"]
    )
    client = setup_mlflow(cfg["evaluate"]["tracking_uri"], logger)
    evaluate(client, cfg, logger)
