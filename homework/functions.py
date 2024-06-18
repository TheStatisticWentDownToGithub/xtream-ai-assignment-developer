import pandas as pd
import numpy as np
from pandas import DataFrame, BooleanDtype
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost
import joblib
import os
from datetime import datetime
import optuna

# Function to load fresh data (e.g., from a CSV file)
def load_data(file_path, nrows=None) -> DataFrame: 
    """
    Loads data from a CSV file into a DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        nrows (int, optional): Number of rows to load. If None, all rows are loaded. Defaults to None.

    Returns:
        DataFrame: The loaded data.
    """
    df = pd.read_csv(file_path) 
    if nrows is not None:
        df = df.head(nrows)
    return df

# Function for data preparation
def data_prep_lm(df: DataFrame, drop_cols=None, to_dummies=None, to_log=None) -> DataFrame:
    """
    Prepares data for linear modeling by dropping columns, converting to dummy variables, and applying logarithms.

    Args:
        df (DataFrame): The input DataFrame.
        drop_cols (list of str, optional): Columns to drop. Defaults to None.
        to_dummies (list of str, optional): Columns to convert to dummy variables. Defaults to None.
        to_log (list of str, optional): Columns to apply logarithm to. Defaults to None.

    Returns:
        DataFrame: The prepared DataFrame.
    """
    # Drop columns if specified
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    
    # Convert specified columns to dummy variables
    if to_dummies is not None:
        df = pd.get_dummies(df, columns=to_dummies, drop_first=True)
        
    # Apply logarithm to specified columns
    if to_log is not None:
        for column in to_log:
            name = 'log_' + column
            df[name] = np.log(df[column])
    
    return df

def data_prep_xgb(df: DataFrame) -> DataFrame:
    """
    Prepares data for XGBoost by converting specific columns to ordered categorical types.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The processed DataFrame.
    """
    # Copy the DataFrame to avoid modifying it directly
    df_processed = df.copy()
    
    # Convert 'cut', 'color', and 'clarity' columns to ordered categories
    df_processed['cut'] = pd.Categorical(df_processed['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    df_processed['color'] = pd.Categorical(df_processed['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    df_processed['clarity'] = pd.Categorical(df_processed['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    
    return df_processed

# Function to train and evaluate the model, and save metrics
def train_and_evaluate_model(
        model_name, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        model_dir, 
        model_history_file,
        target_log_LM: BooleanDtype,
        grid_search_XGB: None,
        random_value = 42):
    """
    Trains and evaluates a model, and saves the model and its metrics.

    Args:
        model_name (str): Name of the model to train ('linear_regression' or 'xgboost').
        X_train (DataFrame): Training features.
        X_test (DataFrame): Test features.
        y_train (Series): Training target.
        y_test (Series): Test target.
        model_dir (str): Directory to save the model.
        model_history_file (str): File to save model history and metrics.
        target_log_LM (BooleanDtype): If True, apply log transformation to the target for linear regression.
        grid_search_XGB (None): If True, perform hyperparameter optimization for XGBoost.
        random_value (int): set seed value

    Raises:
        ValueError: If the specified model_name is not supported.

    Returns:
        None
    """
    # Estimate models
    # LM
    if model_name == 'linear_regression':
        if target_log_LM == True:
            model_name = 'log_linear_regression'
            y_train_log = np.log(y_train)
            model = LinearRegression()
            model.fit(X_train, y_train_log)
            pred_log = model.predict(X_test)
            y_pred = np.exp(pred_log)
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    # XGB
    elif model_name == 'xgboost':
        if grid_search_XGB is True:
            model_name = 'opt_xgboost'
            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=5) # Set to 100 for a real scenario
            print("Best hyperparameters: ", study.best_params)
            xgb_opt = xgboost.XGBRegressor(**study.best_params, enable_categorical=True, random_state=random_value)
            xgb_opt.fit(X_train, y_train)
            y_pred = xgb_opt.predict(X_test)
            model = xgb_opt
        else:
            xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=random_value)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            model = xgb
    else:
        raise ValueError(f"Model {model_name} is not supported. Models supported: ['linear_regression', 'xgboost']")
    
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 4)
    print(f"Trained {model_name} with MAE: {mae}, R2: {r2}")

    # Save model and metrics
    save_model_and_metrics(
        model_name, 
        model, 
        model_dir, 
        mae, 
        r2, 
        model_history_file)

# Function to save the model and metrics
def save_model_and_metrics(model_type: str, model, model_dir, mae, r2, model_history_file):
    """
    Saves the trained model and its evaluation metrics.

    Args:
        model_type (str): Type of the model ('linear_regression' or 'xgboost').
        model: The trained model object.
        model_dir (str): Directory to save the model.
        mae (float): Mean Absolute Error of the model.
        r2 (float): R-squared value of the model.
        model_history_file (str): File to save model history and metrics.

    Returns:
        None
    """
    # Create the MODELS subfolder if it doesn't exist
    models_dir = 'homework\models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Create the subfolder based on the model type
    sub_dir = os.path.join(models_dir, f"{model_type}_LM" if model_type == 'linear_regression' else f"{model_type}_XGB")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Generate the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the file name for the model using the timestamp
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(sub_dir, model_filename)
    
    # Save the model to the file system
    joblib.dump(model, model_path)
    
    # Create a DataFrame with the metrics
    metrics = pd.DataFrame({
        'model_type': [model_type],
        'timestamp': [timestamp],
        'mae': [mae],
        'r2': [r2]
    })

    # Save the metrics to the CSV file
    if not os.path.exists(model_history_file):
        metrics.to_csv(model_history_file, index=False)
    else:
        metrics.to_csv(model_history_file, mode='a', header=False, index=False)

# Function for hyperparameter tuning
def objective(
        trial: optuna.trial.Trial,
        X_train_XGB,
        y_train_XGB,
        random_value = 42) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.

    Args:
        trial (optuna.trial.Trial): A single trial of the optimization process.
        X_train_XGB (DataFrame): Training features for XGBoost.
        y_train_XGB (Series): Training target for XGBoost.
        random_value (int): set seed value

    Returns:
        float: Mean Absolute Error of the model with the suggested hyperparameters.
    """
    # Define hyperparameters to tune
    param = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True
    }

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(X_train_XGB, y_train_XGB, test_size=0.2, random_state=random_value)

    # Train the model
    model = xgboost.XGBRegressor(**param)
    model.fit(x_train, y_train)

    # Make predictions
    preds = model.predict(x_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, preds)

    return mae