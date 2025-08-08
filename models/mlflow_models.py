from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Step 1: Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create a simple model (e.g., Logistic Regression) and start a MLflow experiment
import mlflow
from mlflow import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("classification_experiment")

with mlflow.start_run():
    # Train the model
    model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log metrics
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

# Step 4: Tuning the model with hyperparameter optimization
params_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

for param in params_grid:
    with mlflow.start_run():
        # Train the model with the current parameter
        model = LogisticRegression(C=param, solver='liblinear', random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log metrics
        mlflow.log_param("C", param)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="model")

#Step 5: Training with Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier

rf_param_grid = [10, 50, 100]

for param in rf_param_grid:
    with mlflow.start_run():
        # Train the Random Forest model with the current parameter
        rf_model = RandomForestClassifier(n_estimators=param, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        rf_predictions = rf_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, rf_predictions)
        f1 = f1_score(y_test, rf_predictions)

        # Log metrics
        mlflow.log_param("n_estimators", param)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

# Step 6: Training with XGBoost Classifier model
import xgboost as xgb

xgb_param_grid = [10, 50, 100]

for param in xgb_param_grid:
    with mlflow.start_run():
        # Train the XGBoost model with the current parameter
        xgb_model = xgb.XGBClassifier(n_estimators=param, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Make predictions
        xgb_predictions = xgb_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, xgb_predictions)
        f1 = f1_score(y_test, xgb_predictions)

        # Log metrics
        mlflow.log_param("n_estimators", param)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(xgb_model, artifact_path="model")

# Step 7: Register the best model
best_model_run = mlflow.search_runs(order_by=["metrics.accuracy desc"]).iloc[0]
mlflow.register_model(f"runs:/{best_model_run.run_id}/model", "Best_Logistic_Regression_Model")

# Step 8: Save the best model to output directory
import os
import shutil
output_dir = "outputs/"
models_dir = os.path.join(output_dir, "best_model")
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)  # Remove existing directory if it exists
os.makedirs(output_dir, exist_ok=True)
mlflow.sklearn.save_model(model, os.path.join(output_dir, "best_model"))