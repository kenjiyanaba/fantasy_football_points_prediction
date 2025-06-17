import pytest
from src.model_training import train_model, save_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

def test_train_model():
    # Create a synthetic dataset for testing
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, model_type='RandomForest')

    # Check if the model is an instance of RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)

    # Check if the model can predict
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_save_model(tmp_path):
    # Create a synthetic dataset for testing
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, model_type='RandomForest')

    # Save the model
    model_path = os.path.join(tmp_path, "test_model.pkl")
    save_model(model, model_path)

    # Check if the model file is created
    assert os.path.exists(model_path)