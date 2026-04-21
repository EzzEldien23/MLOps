# ITI-MLOps: Complete MLOps Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive MLOps pipeline demonstrating best practices for machine learning workflows, built for the ITI MLOps course. This project covers the full ML lifecycle: data processing, model training with hyperparameter optimization, evaluation, and deployment (both online and batch).

## Learning Objectives

This project teaches students the following MLOps concepts:

- **Data Versioning**: Managing datasets with DVC
- **Experiment Tracking**: Recording and comparing ML experiments with MLflow
- **Model Registry**: Storing and versioning models in MLflow
- **Pipeline Automation**: Automating ML workflows with DVC stages
- **Configuration Management**: Handling parameters with Hydra and YAML
- **Online Inference**: Serving models for real-time predictions with LitServe
- **Batch Inference**: Running scheduled predictions with Prefect and DuckDB
- **Hyperparameter Optimization**: Bayesian optimization with Hyperopt

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ITI-MLOps Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Raw Data   │───▶│  Processing  │───▶│  Train/Test  │     │
│  │   (CSV)      │    │   (DVC)      │    │  (Parquet)   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                   │            │
│                                                   ▼            │
│                          ┌─────────────────────────────────┐   │
│                          │         Model Training          │   │
│                          │  • Hyperopt (Bayesian Opt)      │   │
│                          │  • MLflow (Tracking/Registry)   │   │
│                          └───────────────┬─────────────────┘   │
│                                          │                     │
│                                          ▼                     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              Model Deployment                         │     │
│  │  ┌────────────────────┐  ┌─────────────────────┐    │     │
│  │  │  Online (LitServe) │  │  Batch (Prefect)    │    │     │
│  │  │  Real-time API     │  │  DuckDB Workflow    │    │     │
│  │  └────────────────────┘  └─────────────────────┘    │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before running this project, ensure you have:

- **Python 3.11+**
- **uv** (package manager)
- **DVC** (data version control)
- **MLflow** (experiment tracking)
- **DuckDB/MotherDuck** (for batch deployment)

Install dependencies:

```bash
uv sync
```

## Project Structure

```
ITI-MLOps/
├── conf/                    # Configuration files
│   ├── config.yaml           # Main config (Hydra defaults)
│   └── pipeline/            # Pipeline configurations
├── data/                    # Data directory
│   ├── external/            # External data sources
│   ├── interim/            # Intermediate transformed data
│   ├── processed/          # Train/test datasets
│   └── raw/                # Raw immutable data
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks (wandb tutorials)
├── reports/                # Evaluation reports
│   └── fake/               # Model evaluation results
├── src/                    # Source code
│   ├── deployment/         # Deployment code
│   │   ├── online/         # LitServe API
│   │   └── batch/          # Prefect workflows
│   ├── fake/               # Custom estimator
│   └── training/           # Training scripts
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Pipeline parameters
├── Makefile                # Convenience commands
└── pyproject.toml          # Project configuration
```

## Pipeline Stages

### 1. Data Processing

Splits the raw Iris dataset into training and test sets.

```bash
dvc repro process_data
```

**Input**: `data/raw/Iris.csv`  
**Output**: `data/processed/Iris-train.parquet`, `data/processed/Iris-test.parquet`

### 2. Model Training

Trains a classification model using Bayesian hyperparameter optimization.

```bash
dvc repro train
```

**Features**:
- Custom `FakeEstimator` (random classifier based on sklearn)
- Hyperopt for hyperparameter tuning (TPE algorithm)
- MLflow experiment tracking
- Model registration to MLflow registry

**Input**: Train/test parquet files  
**Output**: `models/fake/final_model.pkl`, `models/fake/model_target_translator.pkl`

### 3. Model Evaluation

Evaluates the production model and generates a report.

```bash
dvc repro evaluate
```

**Metrics**: Accuracy, Precision, Recall  
**Output**: `reports/fake/evaluation_report.json`

### 4. Online Deployment (LitServe)

Serve the model as a REST API for real-time inference.

```bash
uv run python server.py
```

The API accepts input data and returns predictions. See `src/deployment/online/api.py` for implementation.

### 5. Batch Deployment (Prefect)

Run scheduled batch inference with Prefect workflows.

```bash
uv run python forecast.py
```

Uses Prefect for orchestration and DuckDB/MotherDuck for data storage.

## Quick Start

### Run the Full Pipeline

```bash
# Install dependencies
make requirements

# Process data
dvc repro process_data

# Train model
dvc repro train

# Evaluate model
dvc repro evaluate
```

### Other Commands

```bash
# Create virtual environment
make create_environment

# Lint code
make lint

# Format code
make format

# Clean cache
make clean
```

## Tools Reference

| Tool | Purpose |
|------|---------|
| **DVC** | Data and model version control |
| **MLflow** | Experiment tracking and model registry |
| **Hydra** | Configuration management |
| **Hyperopt** | Bayesian hyperparameter optimization |
| **LitServe** | Model serving for online inference |
| **Prefect** | Workflow orchestration for batch inference |
| **DuckDB** | Embedded analytical database |
| **Dagshub** | ML experiment hosting platform |
| **wandb** (optional) | Weights & Biases logging |

## Configuration

Key parameters are defined in `params.yaml`:

```yaml
seed: 42
data:
  raw_data_path: data/raw
  processed_data_path: data/processed
  file_name: Iris
  target_column: Species
  test_size: 0.15

model:
  model_name: fake
  optimization_params:
    n_folds: 5
    max_evals: 10
    scoring: accuracy
```

## Expected Outputs

After running the pipeline, you should have:

- Processed datasets in Parquet format
- Trained model saved as pickle files
- MLflow experiment with logged parameters and metrics
- Model registered in MLflow registry (Staging/Production)
- Evaluation report with accuracy, precision, recall
- Working API for online inference
- Prefect workflow for batch forecasting

## License

MIT License - See LICENSE file for details.
