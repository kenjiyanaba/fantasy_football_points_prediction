import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    evaluation_results = {
        "Mean Squared Error": mse,
        "R^2 Score": r2
    }
    
    return evaluation_results

def print_evaluation_results(results):
    print("\nModel Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")