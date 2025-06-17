import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_average_fantasy_points(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="position", y="fantasy_points_ppr", estimator='mean')
    plt.title("Average Fantasy Points (PPR) by Position")
    plt.xticks(rotation=45)
    plt.ylabel("Fantasy Points (PPR)")
    plt.show()

def plot_fantasy_points_by_age(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="age", y="fantasy_points_ppr", hue="position", alpha=0.6)
    plt.title("Fantasy Points by Age")
    plt.xlabel("Age")
    plt.ylabel("Fantasy Points (PPR)")
    plt.show()

def plot_feature_importance(importances, feature_names):
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature")
    plt.title("Feature Importance in Fantasy Points Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.title("Actual vs Predicted Fantasy Points")
    plt.xlabel("Actual Fantasy Points")
    plt.ylabel("Predicted Fantasy Points")
    plt.tight_layout()
    plt.show()