from prefect import task, flow
from prefect.tasks import task_input_hash
import duckdb
import pandas as pd
import datetime


@task(
    name="GetDailyData",
    description="get daily weather data from hourly data as dataframe",
    tags=["Get", "DailyData"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_daily_data(conn, running_date: str, logger) -> pd.DataFrame:
    """Get the daily data from hourly weather data

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Get Daily Weather Data of {running_date}")
    df_daily = conn.sql(f"""
                        SELECT location_id,
                               strftime(reading_timestamp, '%Y-%m-%d') AS reading_date,
                               AVG(temperature) AS temperature,
                        FROM ml_apps.iti_weather_forecasting.hourly_weather_data
                        WHERE strftime(reading_timestamp, '%Y-%m-%d') = '{running_date}'
                        GROUP BY location_id, reading_date
                        """).df()
    return df_daily


@task(
    name="CheckIfDailyExists",
    description=(
        "Check the MotherDuck if the Data of the passed date exists, and if it"
        " removes it"
    ),
    tags=["Check", "DailyData", "DB"],
    retry_delay_seconds=30,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def check_if_data_exists(conn, running_date: str, logger) -> None:
    """Check the MotherDuck if the Data of the passed date exists,
      and if it removes it

    Parameters
    ----------
    conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Check if Data Exist for the current-date: {running_date}")
    daily_df = conn.sql(f"""
                       SELECT * FROM ml_apps.iti_weather_forecasting.daily_weather_data 
                        WHERE strftime(reading_date, '%Y-%m-%d') = '{running_date}'
                    """).df()
    logger.info(f"found {len(daily_df)} records")
    if len(daily_df) > 0:
        logger.info("data found, and will be removed safely")
        conn.sql(f"""
                DELETE FROM ml_apps.iti_weather_forecasting.daily_weather_data 
                WHERE strftime(reading_date, '%Y-%m-%d') = '{running_date}'
            """)
        logger.info("data removed successfully")
    else:
        logger.info(f"no data exists for {running_date}")


@task(
    name="LoadToDailyMotherDuck",
    description="Load Data into MotherDuck Database",
    tags=["load", "updated-data", "motherduck"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def load_to_motherduck(df: pd.DataFrame, conn, logger) -> None:
    """Load Data into Database

    Parameters
    ----------
    df : pd.DataFrame
        API DataFrame
    conn : motherduck database connection
    """
    conn.sql(
        "INSERT INTO ml_apps.iti_weather_forecasting.daily_weather_data SELECT * FROM df"
    )
    logger.info("Data Insert Successfully into daily_weather_data Table")


@task(
    name="DeleteOutofRangeDailyData",
    description="Delete Out of Range Data to mintain Staorage Space",
    tags=["delete", "out-of-range-data", "motherduck", "storage"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def delete_out_of_range_data(conn, thresh_date: str, logger) -> None:
    logger.info("Deleting Out of Range Data")
    conn.sql(f"""
            DELETE FROM ml_apps.iti_weather_forecasting.daily_weather_data 
            WHERE reading_date <= CAST('{thresh_date}' AS DATE)-2000
        """)
    logger.info("Data Deleted Successfully")


@flow(
    name="InferenceDataPreparationFlow",
    description="Prepare data to be daily, and insert it to MotherDuck",
    validate_parameters=True,
    log_prints=True,
)
def data_prep_flow(db_token: str, date: str, logger) -> bool:
    """sub-flow of preparing daily data for inference process

    Parameters
    ----------
    db_token: MotherDuck Database Credentials
    date: str
        running date of the process,
        and date used to delete data that exceeds 2000 days from
        this date.
    """
    inference_flag = False
    logger.info("Connecting To MotherDuck to Load Data")
    with duckdb.connect(f"md:?motherduck_token={db_token}") as conn:
        logger.info("Connection Successfully intiated")
        df = get_daily_data(conn=conn, running_date=date, logger=logger)
        if len(df) > 0:
            inference_flag = True
            logger.info(f"Daily Data for {date} Exists")
            check_if_data_exists(conn=conn, running_date=date, logger=logger)
            load_to_motherduck(df=df, conn=conn, logger=logger)
            delete_out_of_range_data(conn=conn, thresh_date=date, logger=logger)
    logger.info("Connection with MotherDuck Closed")
    return inference_flag