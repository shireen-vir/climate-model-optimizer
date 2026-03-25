import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def optimize_model(climate_data, target_variable):
    """
    This function optimizes a climate model by selecting the best features and hyperparameters for a random forest regressor.
    
    Args:
        climate_data (pd.DataFrame): A pandas DataFrame containing climate data.
        target_variable (str): The name of the target variable in the climate data.
        
    Returns:
        tuple: A tuple containing the optimized model, the training data, and the testing data.
    """
    # Split the data into features and target
    X = climate_data.drop(target_variable, axis=1)
    y = climate_data[target_variable]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    
    return model, X_train, X_test, y_train, y_test

def main():
    # Example usage
    climate_data = pd.DataFrame({
        'temperature': np.random.rand(100),
        'humidity': np.random.rand(100),
        'precipitation': np.random.rand(100)
    })
    target_variable = 'temperature'
    model, X_train, X_test, y_train, y_test = optimize_model(climate_data, target_variable)
    print(f'Optimized model: {model}')
    print(f'Training data shape: {X_train.shape}')
    print(f'Testing data shape: {X_test.shape}')

if __name__ == '__main__':
    main()