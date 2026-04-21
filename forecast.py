import argparse
import datetime
import yaml

from prefect import flow
from src.deployment.batch.forecast import forecast_flow
from src.logger import ExecutorLogger
from dotenv import dotenv_values


@flow(
    name="InferenceFlow",
    description="Inference/Monitoring Main Flow",
    validate_parameters=True,
    log_prints=True,
)
def pred_flow(db_token, running_date: str, logger) -> None:
    """main flow of weather forecasting & performance monitoring

    Parameters
    ----------
    db_token : str
        MotherDuck Database Credentials
    running_date : str
        running date of the process
    model_path : str
        path of model's pickle file
    """
    forecast_flow(db_token=db_token, date=running_date, logger=logger)


if __name__ == "__main__":
    logger = ExecutorLogger("batch_scoring")
    ENV = dotenv_values(".env")
    default_date = datetime.datetime.strftime(
        datetime.datetime.now() - datetime.timedelta(days=2), "%Y-%m-%d"
    )
    parser = argparse.ArgumentParser(description="ML Job Parameters")
    parser.add_argument("--running_date", default=default_date, type=str)
    args = parser.parse_args()
    pred_flow(
        db_token=ENV["MOTHERDUCK_TOKEN"],
        running_date=args.running_date,
        logger=logger
    )