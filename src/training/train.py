import os
import pickle
from functools import partial
from typing import Any, Dict

import dagshub
import dvc.api
import numpy as np
import pandas as pd
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import cross_validate
from dotenv import load_dotenv

from src.fake.estimator import FakeEstimator
from src.logger import ExecutorLogger
from src.training.model_wrapper import ModelWrapper


def encode_target_col(
    cfg: Dict[str, Any],
    logger,
):
    train_df = pd.read_parquet(
        os.path.join(
            cfg["model"]["processed_data_path"],
            f"{cfg['model']['file_name']}-train.parquet",
        )
    )
    test_df = pd.read_parquet(
        os.path.join(
            cfg["model"]["processed_data_path"],
            f"{cfg['model']['file_name']}-test.parquet",
        )
    )
    X_train, y_train = (
        train_df.drop(cfg["model"]["target_column"], axis=1),
        train_df[cfg["model"]["target_column"]],
    )
    X_test, y_test = (
        test_df.drop(cfg["model"]["target_column"], axis=1),
        test_df[cfg["model"]["target_column"]],
    )
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    # save the encoder/decoder of target
    label_translator = {"encoder": encoder, "decoder": decoder}
    logger.info("encoder/decoder of target created successfully")
    if not os.path.exists(
        os.path.join(cfg["model"]["model_path"], cfg["model"]["model_name"])
    ):
        os.makedirs(
            os.path.join(cfg["model"]["model_path"], cfg["model"]["model_name"])
        )
    with open(
        os.path.join(
            cfg["model"]["model_path"],
            cfg["model"]["model_name"],
            "model_target_translator.pkl",
        ),
        "wb",
    ) as pkl:
        pickle.dump(label_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def objective(params: Dict[str, Any], X, y, n_folds: int) -> Dict[str, Any]:
    model = FakeEstimator(**params)
    scores = cross_validate(model, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
    score = np.mean(scores["test_score"])
    return {"loss": score, "params": params, "status": STATUS_OK}


def setup_mlflow(tracking_uri: str, logger):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    logger.info("MLFlow Client Defined and tracking URI Setted Successfully.")
    return client


def trainer(X, y, cfg: Dict[str, Any], logger):
    SPACE = {
        cfg["model"]["optimization_params"]["hyperparameter_search"]["random_state"][
            "name"
        ]: scope.int(
            hp.quniform(
                cfg["model"]["optimization_params"]["hyperparameter_search"][
                    "random_state"
                ]["name"],
                cfg["model"]["optimization_params"]["hyperparameter_search"][
                    "random_state"
                ]["min"],
                cfg["model"]["optimization_params"]["hyperparameter_search"][
                    "random_state"
                ]["max"],
                cfg["model"]["optimization_params"]["hyperparameter_search"][
                    "random_state"
                ]["step"],
            )
        ),
    }
    logger.info("Load encoder/decoder of target variable")
    with open(
        os.path.join(
            cfg["model"]["model_path"],
            cfg["model"]["model_name"],
            "model_target_translator.pkl",
        ),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    bayes_trials = Trials()
    fmin_objective = partial(
        objective,
        X=X,
        y=y_train_enc,
        n_folds=cfg["model"]["optimization_params"]["n_folds"],
    )
    logger.info("optimization started")
    fmin(
        fn=fmin_objective,
        space=SPACE,
        algo=tpe.suggest,
        max_evals=cfg["model"]["optimization_params"]["max_evals"],
        trials=bayes_trials,
    )
    logger.info("optimization completed")
    best_model = bayes_trials.results[
        np.argmin([r["loss"] for r in bayes_trials.results])
    ]
    params = best_model["params"]
    with mlflow.start_run():
        mlflow.autolog()
        final_model = FakeEstimator(**params)
        final_model.fit(X, y_train_enc)
        logger.info("save the final optimized model")
        if not os.path.exists(
            os.path.join(cfg["model"]["model_path"], cfg["model"]["model_name"])
        ):
            os.makedirs(
                os.path.join(
                    cfg["model"]["model_path"], 
                    cfg["model"]["model_name"]
                )
            )
        with open(
            os.path.join(
                cfg["model"]["model_path"], 
                cfg["model"]["model_name"], 
                "final_model.pkl"
            ),
            "wb",
        ) as pkl:
            pickle.dump(final_model, pkl)
        logger.info("model trained and saved successfully")
        run_id = mlflow.active_run().info.run_id
        train_preds = final_model.predict(X)
        signature = mlflow.models.infer_signature(X, train_preds)
        mlflow.pyfunc.log_model(
            cfg["model"]["model_name"], 
            python_model=ModelWrapper(),
            artifacts={ 
                'encoder': os.path.join(
                    cfg["model"]["model_path"],
                    cfg["model"]["model_name"],
                    "model_target_translator.pkl",
                ),
                'model': os.path.join(
                    cfg["model"]["model_path"], 
                    cfg["model"]["model_name"], 
                    "final_model.pkl"
                )
            },
            signature=signature,
            registered_model_name=cfg["model"]["model_name"],
        )
        mlflow.log_params(params)
        mlflow.log_metrics({
            f"cv_{cfg['model']['optimization_params']['scoring']}_score": best_model["loss"]
        })
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"

        model_details = mlflow.register_model(
            model_uri=model_uri, 
            name=cfg["model"]["model_name"]
        )
        logger.info("Model registered successfully!!")

        return model_details, run_id


def move_model_to_prod(client, model_details, logger) -> None:
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )
    logger.info("Model transitioned to prod stage")


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    load_dotenv(".env")
    cfg = dvc.api.params_show()
    logger.info(
        "Paramsters: \n"
        f"{cfg['model']}"
    )
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"), 
        repo_name=cfg["model"]["repo_name"], 
        mlflow=cfg["model"]["use_mlflow"]
    )
    client = setup_mlflow(cfg["model"]["tracking_uri"], logger)
    X_train, y_train, X_test, y_test = encode_target_col(cfg, logger)
    model_details, run_id = trainer(X_train, y_train, cfg, logger)
    move_model_to_prod(client, model_details, logger)
