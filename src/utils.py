import pandas as pd

TARGET = "Survived"
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

DEFAULT_TRAIN_PATH = "data/raw/train.csv"
DEFAULT_TEST_PATH = "data/raw/test.csv"
DEFAULT_SUBMISSION_PATH = "data/raw/gender_submission.csv"


def load_data(
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    submission_path: str = DEFAULT_SUBMISSION_PATH,
):

    print(" Loading data...")


    train_df = pd.read_csv(train_path)

    if TARGET not in train_df.columns:
        raise ValueError(" 'Survived' not found in train data")

    X_train = train_df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
    y_train = train_df[TARGET]

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")

 
    submission_df = pd.read_csv(submission_path)

    if TARGET not in submission_df.columns:
        raise ValueError(" 'Survived' not found in submission file")

  
    test_with_target = test_df.merge(submission_df, on="PassengerId", how="left")
    y_test = test_with_target[TARGET]


    if y_test.isnull().sum() > 0:
        raise ValueError(" Missing values in y_test after merge")

    print(f"✅ Train shape: {X_train.shape}")
    print(f"✅ Test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test