import pytest
import pandas as pd
from src.model_evaluation import evaluate_model

def test_evaluate_model():
    # Sample data for testing
    y_true = pd.Series([10, 20, 30, 40, 50])
    y_pred = pd.Series([12, 18, 29, 37, 55])
    
    mse, r2 = evaluate_model(y_true, y_pred)
    
    # Check if the mean squared error is calculated correctly
    expected_mse = ((y_true - y_pred) ** 2).mean()
    assert mse == pytest.approx(expected_mse, rel=1e-2)
    
    # Check if the R^2 score is calculated correctly
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    expected_r2 = 1 - (ss_residual / ss_total)
    assert r2 == pytest.approx(expected_r2, rel=1e-2)