import requests
import duckdb
import pandas as pd
import datetime

from prefect import task, flow
from prefect.tasks import task_input_hash

from src.deployment.batch.data_models import URLParams, APIData
from src.logger import ExecutorLogger


@task(
    name="GetAPIData",
    description=(
        "access meteo-weather API to get the historical hourly weather data needed"
    ),
    tags=["raw-data", "weather-data", "get"],
    cache_key_fn=task_input_hash,
    cache_expiration=datetime.timedelta(minutes=10),
    retry_delay_seconds=120,
    retries=3,
    log_prints=True,
    timeout_seconds=60,
)
def get_api_data(data_url: str, params: URLParams, logger) -> APIData:
    """get weather data for specific date from URL,
    the passed datetime parameter, will get the data for previous two days.

    Parameters
    ----------
    data_url : str
        url of weather data provider
    params : Pydantic-BaseModel
        parameters of weather API

    Returns
    -------
    APIData
        JSON data of weather API
    """
    logger.info("Retrieve Weather Data From API")
    response = requests.get(url=data_url, params=params)
    if response.status_code == 200:
        logger.info(f"Data Retrieved Successfully with status code: {response.status_code}")
        data = response.json()
        checking_lst = data["hourly"]["temperature_2m"]
        if len(checking_lst) >= 24 and None not in checking_lst:
            logger.info("No Missing or in-complete records")
            return data
        else:
            raise ValueError("Retrieved Data is in-complete or Missing")
    else:
        raise ConnectionError(
            f"error when retrieving the data, status-code: {response.status_code}"
        )


@task(
    name="TransformToDataFrame",
    description="Convert API Data to DataFrame, and Prepare to load into MotherDuck",
    tags=["transform", "prepare-api-data"],
    retry_delay_seconds=30,
    retries=2,
    log_prints=True,
    timeout_seconds=60,
)
def transform_api_data(data, logger) -> pd.DataFrame:
    """Transform Data to be like Database-Schema Shape

    Parameters
    ----------
    data : APIData
        JSON API Data

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of API Data
    """
    logger.info("Transform Data to Database Schema like")
    logger.info("Get Location Data of API Data")
    df = pd.DataFrame({
        "location_id": [75354428 for _ in range(len(data["hourly"]["time"]))],
        "reading_timestamp": data["hourly"]["time"],
        "temprature": data["hourly"]["temperature_2m"],
        "tz": [data["timezone"] for _ in range(len(data["hourly"]["time"]))],
    })
    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    logger.info("Data Transformed Successfully, and Ready for Loading into DB")
    return df


@task(
    name="CheckIfDataExist",
    description=(
        "Check the MotherDuck if the Data of the passed date exists, and if it"
        " removes it"
    ),
    tags=["Quality", "Check", "motherduck"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=120,
)
def check_if_data_exists(db_conn, running_dt: str, logger) -> None:
    """Check the MotherDuck if the Data of the passed date exists,
      and if it removes it

    Parameters
    ----------
    db_conn : MotherDuck Database Connection
    running_dt: str
         string format of pipeline running date
    """
    logger.info(f"Check if Data Exist for the current-date: {running_dt}")
    db_conn.sql("USE ml_apps")
    data_df = db_conn.sql(f"""
                          SELECT * FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
                          WHERE strftime(reading_timestamp, '%Y-%m-%d') = '{running_dt}'
                        """).df()
    logger.info(f"found {len(data_df)} records")
    if len(data_df) > 0:
        logger.info("data found, and will be removed safely")
        db_conn.sql(f"""
                    DELETE FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
                    WHERE strftime(reading_timestamp, '%Y-%m-%d') = '{running_dt}'
                """)
        logger.info("data removed successfully")


@task(
    name="LoadToMotherDuck",
    description="Load Data into MotherDuck Database",
    tags=["load", "updated-data", "motherduck"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=120,
)
def load_to_motherduck(df: pd.DataFrame, db_conn, logger) -> None:
    """Load Data into Database

    Parameters
    ----------
    df : pd.DataFrame
        API DataFrame
    db_conn : motherduck database connection
    """
    db_conn.sql("USE ml_apps")
    db_conn.sql(
        "INSERT INTO ml_apps.iti_weather_forecasting.hourly_weather_data SELECT * FROM df"
    )
    logger.info("Data Insert Successfully into hourly_weather_data Table")


@task(
    name="DeleteOutofRangeData",
    description="Delete Out of Range Data to mintain Staorage Space",
    tags=["delete", "out-of-range-data", "motherduck", "storage"],
    retry_delay_seconds=60,
    retries=3,
    log_prints=True,
    timeout_seconds=120,
)
def delete_out_of_range_data(db_conn, thresh_dt: str, logger) -> None:
    logger.info("Deleting Out of Range Data")
    db_conn.sql("USE ml_apps")
    db_conn.sql(f"""
                DELETE FROM ml_apps.iti_weather_forecasting.hourly_weather_data 
                WHERE reading_timestamp <= CAST('{thresh_dt}' AS DATE)-800
        """)
    logger.info("Data Deleted Successfully")


@flow(
    name="DataFlow",
    description=(
        "Manages the execution flow of getting the data from API and saving it to"
        " MotherDuck."
    ),
    validate_parameters=True,
    log_prints=True,
)
def data_flow(
    api_data_url: str, 
    url_params: URLParams, 
    db_token: str, 
    deleting_thresh_dt: str,
    logger,
) -> None:
    """sub-flow of raw-data that get data from api,
    and load it into motherduck database..

    Parameters
    ----------
    api_data_url : str
        weather data API url
    url_params : URLParams
        parameters needed to get the API data
    db_token: MotherDuck Database Credentials
    deleting_thresh_dt: str
        date used to delete data that exceeds 800 days from
        this date.
    """
    api_data = get_api_data(data_url=api_data_url, params=url_params, logger=logger)
    api_df = transform_api_data(data=api_data, logger=logger)
    logger.info("Connecting To MotherDuck to Load Data")
    with duckdb.connect(f"md:?motherduck_token={db_token}") as conn:
        logger.info("Connection Successfully intiated")
        check_if_data_exists(db_conn=conn, running_dt=deleting_thresh_dt, logger=logger)
        load_to_motherduck(df=api_df, db_conn=conn, logger=logger)
        delete_out_of_range_data(db_conn=conn, thresh_dt=deleting_thresh_dt, logger=logger)
    logger.info("Connection with MotherDuck Closed")