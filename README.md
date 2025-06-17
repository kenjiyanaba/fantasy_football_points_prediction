# Fantasy Points Prediction Project

This project aims to predict fantasy points per fantasy_points_ppr for NFL players using machine learning models. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization of results.

# Project Structure

```
fantasy-points-prediction
├── data
│   ├── raw
│   │   └── fantasy_stats.csv        # Raw fantasy statistics data
│   └── processed                     # Directory for processed datasets
├── notebooks
│   └── exploratory_analysis.ipynb    # Jupyter notebook for exploratory data analysis
├── src
│   ├── data_preprocessing.py         # Functions for loading and cleaning data
│   ├── feature_engineering.py        # Functions for creating new features
│   ├── model_training.py             # Functions for training machine learning models
│   ├── model_evaluation.py           # Functions for evaluating model performance
│   └── visualization.py               # Functions for visualizing results
├── tests
│   ├── test_data_preprocessing.py    # Unit tests for data preprocessing
│   ├── test_feature_engineering.py    # Unit tests for feature engineering
│   ├── test_model_training.py        # Unit tests for model training
│   └── test_model_evaluation.py      # Unit tests for model evaluation
├── requirements.txt                  # Required Python packages
├── .gitignore                        # Files and directories to ignore in version control
└── README.md                         # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fantasy-points-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place the raw fantasy statistics data in the `data/raw` directory.

## Usage Guidelines

- Use the `notebooks/exploratory_analysis.ipynb` for initial data exploration and visualization.
- Run the scripts in the `src` directory to preprocess data, engineer features, train models, and evaluate their performance.
- Utilize the `tests` directory to ensure the functionality of the code through unit tests.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.