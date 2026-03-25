import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    """
    Climate Model Optimizer Tool

    This script is designed to optimize climate models by selecting the most relevant features and tuning hyperparameters.
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Load the climate dataset
    climate_data = pd.read_csv('climate_data.csv')

    # Split the data into training and testing sets
    X = climate_data.drop('target', axis=1)
    y = climate_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a random forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Performance (MSE): {mse}")

    # Tune hyperparameters to optimize the model
    from sklearn.model_selection import GridSearchCV
    param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 15]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the optimized hyperparameters and the corresponding model performance
    print(f"Optimized Hyperparameters: {grid_search.best_params_}")
    print(f"Optimized Model Performance (MSE): {grid_search.best_score_}")

if __name__ == "__main__":
    main()