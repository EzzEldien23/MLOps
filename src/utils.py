import pandas as pd
import os

TARGET = "Survived"
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

DEFAULT_TRAIN_PATH = "data/raw/train.csv"
DEFAULT_TEST_PATH = "data/raw/test.csv"
DEFAULT_SUBMISSION_PATH = "data/raw/gender_submission.csv"


def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_git_commit() -> str:
    """Retrieves the current git commit hash."""
    import git
    try:
        repo = git.Repo(get_project_root(), search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        print(f"[WARNING] Failed to retrieve git commit: {e}")
        return "unknown"


def get_dataset_version() -> str:
    """Retrieves the dataset version MD5 hash from dvc.lock."""
    import yaml
    root = get_project_root()
    dvc_lock_path = os.path.join(root, "dvc.lock")
    if not os.path.exists(dvc_lock_path):
        return "unknown"
    try:
        with open(dvc_lock_path, "r") as f:
            lock_data = yaml.safe_load(f)
        stages = lock_data.get("stages", {})
        train_stage = stages.get("train", {})
        deps = train_stage.get("deps", [])
        for dep in deps:
            if dep.get("path") == "data/raw":
                return dep.get("md5", "unknown")
    except Exception as e:
        print(f"[WARNING] Failed to parse {dvc_lock_path}: {e}")
    return "unknown"


def setup_mlflow_tracking() -> bool:
    """Configures MLflow tracking URI and credentials for DagsHub with local fallback."""
    import mlflow
    import dotenv

    root = get_project_root()
    dotenv_path = os.path.join(root, ".env")
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)

    # Allow local file store tracking backend in MLflow 3.x+
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    owner = os.environ.get("DAGSHUB_REPO_OWNER")
    repo = os.environ.get("DAGSHUB_REPO_NAME")
    token = os.environ.get("DAGSHUB_TOKEN")

    if not owner or not repo:
        print("[WARNING] DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME not set. Falling back to local MLflow tracking.")
        local_uri = f"file:///{os.path.join(root, 'mlruns').replace(os.sep, '/')}"
        mlflow.set_tracking_uri(local_uri)
        return False

    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    print(f" Connecting to DagsHub MLflow tracking at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        print("[INFO] DagsHub credentials configured via DAGSHUB_TOKEN.")
    else:
        print("[WARNING] DAGSHUB_TOKEN not set. Connection might fail if the repository is private.")
    
    try:
        # Check connection by calling a lightweight MLflow API
        mlflow.search_experiments(max_results=1)
        print("[OK] Successfully connected to DagsHub MLflow server.")
        return True
    except Exception as e:
        print(f"[ERROR] DagsHub MLflow connection check failed: {e}")
        print("[INFO] Falling back to local MLflow tracking...")
        local_uri = f"file:///{os.path.join(root, 'mlruns').replace(os.sep, '/')}"
        mlflow.set_tracking_uri(local_uri)
        return False


def load_data(
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    submission_path: str = DEFAULT_SUBMISSION_PATH,
):

    print(" Loading data...")

    root = get_project_root()
    if not os.path.isabs(train_path):
        train_path = os.path.join(root, train_path)
    if not os.path.isabs(test_path):
        test_path = os.path.join(root, test_path)
    if not os.path.isabs(submission_path):
        submission_path = os.path.join(root, submission_path)

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

    print(f"[OK] Train shape: {X_train.shape}")
    print(f"[OK] Test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test

