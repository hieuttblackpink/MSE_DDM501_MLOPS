# MLflow Classification Project

This project demonstrates a complete ML workflow using MLflow, scikit-learn, XGBoost, and Flask. It includes model training, experiment tracking, model serving via a web API, and Docker-based deployment.

## Project Structure

```
.
├── app/
│   ├── app.py                # Flask API for model inference
│   └── template/
│       └── index.html        # Web UI for predictions
├── models/
│   └── mlflow_models.py      # Model training, MLflow logging
├── outputs/
│   └── best_model/           # Saved best model and environment files
├── docker-compose.yml        # Orchestrates all services
├── dockerfile.app            # Dockerfile for Flask API
├── dockerfile.mlflowui       # Dockerfile for MLflow UI
├── dockerfile.train          # Dockerfile for training job
├── requirements.txt          # Python dependencies
├── .gitlab-ci.yml            # GitLab CI/CD pipeline
├── .gitignore
└── README.md
```

## Features

- **MLflow Tracking**: Logs experiments, parameters, metrics, and models.
- **Model Training**: Trains Logistic Regression, Random Forest, and XGBoost classifiers with hyperparameter tuning.
- **Model Registry**: Registers the best model in MLflow.
- **Model Serving**: Flask API for real-time predictions.
- **Web UI**: Simple HTML interface for predictions.
- **Dockerized**: All components run in containers for easy deployment.
- **CI/CD**: Automated build and test pipeline with GitLab CI.

---

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- (Optional) Python 3.13.5+ if running locally

---

## Quick Start

### 1. Clone the Repository

```sh
git clone https://gitlab.com/hieu24mse23185/MSE_DDM501_MLOPS
cd MSE_DDM501_MLOPS
```

### 2. Build and Run with Docker Compose

```sh
docker-compose up --build
```

This will start:
- **MLflow UI** at [http://localhost:5000](http://localhost:5000)
- **Flask API** at [http://localhost:5050](http://localhost:5050)
- **Model Training** (runs once to train and log models)

### 3. Access the Web UI

Open [http://localhost:5050](http://localhost:5050) in your browser.  
Enter 20 comma-separated feature values to get a prediction.

---

## Detailed Instructions

### Model Training

- The training script [`models/mlflow_models.py`](models/mlflow_models.py) generates synthetic data, trains multiple models, logs them to MLflow, and saves the best model to [`outputs/best_model/`](outputs/best_model/).
- Training runs automatically via Docker Compose or can be run manually:

```sh
docker build -f dockerfile.train -t mlflow-train .
docker run --rm -v $(pwd)/outputs:/app/outputs mlflow-train
```

### MLflow Tracking UI

- Access at [http://localhost:5000](http://localhost:5000)
- View experiments, runs, parameters, metrics, and artifacts.

### Flask API

- Serves predictions using the best trained model.
- Endpoint: `POST /predict`
- Example request:

```json
POST http://localhost:5050/predict
Content-Type: application/json

{
  "features": [0.1, 0.2, ..., 2.0]  // 20 numeric values
}
```

- Example response:

```json
{
  "prediction": 1,
  "probability": [0.12, 0.88]
}
```

### Web UI

- User-friendly interface at [http://localhost:5050](http://localhost:5050)
- Enter 20 features, get prediction and probability.

---

## Development

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run Training Locally

```sh
python models/mlflow_models.py
```

### Run Flask API Locally

```sh
python app/app.py
```

---

## CI/CD

- The pipeline in [`.gitlab-ci.yml`](.gitlab-ci.yml) builds Docker images and runs integration tests on the main branch.

---

## File Descriptions

- [`models/mlflow_models.py`](models/mlflow_models.py): Model training, MLflow logging, and model export.
- [`app/app.py`](app/app.py): Flask API for serving predictions.
- [`app/template/index.html`](app/template/index.html): Web UI for user input.
- [`outputs/best_model/`](outputs/best_model/): Saved model and environment files.
- [`docker-compose.yml`](docker-compose.yml): Multi-container orchestration.
- [`dockerfile.*`](dockerfile.app): Dockerfiles for each service.

---

## Notes

- The project uses synthetic data for demonstration.
- Adjust model code and data as needed for your use case.
- Ensure Docker volumes (`mlruns`, `outputs`) are writable.

---

## Authors

- Tran Trong Hieu