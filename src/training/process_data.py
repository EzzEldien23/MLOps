import os
from typing import Any, Dict

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import ExecutorLogger


def read_process_data(
    cfg: Dict[str, Any],
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(
        os.path.join(cfg["data"]["raw_data_path"], f"{cfg['data']['file_name']}.csv")
    )
    df.set_index(cfg["data"]["id_column"], inplace=True)
    train_df, test_df = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=df[cfg["data"]["target_column"]],
    )
    train_df.to_parquet(
        os.path.join(
            cfg["data"]["processed_data_path"],
            f"{cfg['data']['file_name']}-train.parquet",
        ),
        engine="pyarrow",
    )
    test_df.to_parquet(
        os.path.join(
            cfg["data"]["processed_data_path"],
            f"{cfg['data']['file_name']}-test.parquet",
        ),
        engine="pyarrow",
    )


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    cfg = dvc.api.params_show()
    logger.info(
        "Paramsters: \n"
        f"{cfg['data']}"
    )
    read_process_data(cfg, logger)
