import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flaml import AutoML

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AutoML
automl = AutoML()

# Start an MLflow run
with mlflow.start_run():
    # Define AutoML settings
    automl_settings = {
        "time_budget": 600,  # time budget in seconds
        "metric": 'accuracy',  # evaluation metric
        "task": 'classification',  # task type
        "log_file_name": "flaml.log",  # log file
    }
    
    # Train AutoML
    automl.fit(X_train, y_train, **automl_settings)
    
    # Predict and evaluate the best model
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters, metrics, and the best model
    mlflow.log_param("model_type", "FLAML")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(automl.model, "best_model")

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Log the best configuration found by FLAML
    best_config = automl.best_config
    for key, value in best_config.items():
        mlflow.log_param(key, value)
