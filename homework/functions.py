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


# Funzione per caricare i dati freschi (ad esempio da un file CSV)
def load_data(file_path, nrows=None) -> DataFrame: 
    df = pd.read_csv(file_path) 
    if nrows is not None:
        df = df.head(nrows)
    return df

# Funzione per la data preparation
def data_prep_lm(df: DataFrame, drop_cols=None, to_dummies=None, to_log=None) -> DataFrame:
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

def data_prep_xgb(df):
    # Copia il DataFrame in modo da non modificarlo direttamente
    df_processed = df.copy()
    
    # Converti le colonne 'cut', 'color' e 'clarity' in categorie ordinate
    df_processed['cut'] = pd.Categorical(df_processed['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    df_processed['color'] = pd.Categorical(df_processed['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    df_processed['clarity'] = pd.Categorical(df_processed['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    
    return df_processed

# Funzione per addestrare il modello e salvare le metriche
def train_and_evaluate_model(
        model_name, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        model_dir, 
        model_history_file,
        target_log_LM: BooleanDtype,
        grid_search_XGB: None):
    # Stima modelli
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
            # study.optimize(objective(trial = optuna.trial.Trial, X_train_XGB = X_train, y_train_XGB = y_train), n_trials=100)
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=5) # imposta 100
            print("Best hyperparameters: ", study.best_params)
            xgb_opt = xgboost.XGBRegressor(**study.best_params, enable_categorical=True, random_state=42)
            xgb_opt.fit(X_train, y_train)
            y_pred = xgb_opt.predict(X_test)
            model = xgb_opt
        else:
            xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            model = xgb
    else:
        raise ValueError(f"Model {model_name} is not supported. Models supported: ['linear_regression', 'xgboost']")
    
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 4)
    print(f"Trained {model_name} with MAE: {mae}, R2: {r2}")

    # Salvataggio modelli
    save_model_and_metrics(
        model_name, 
        model, 
        model_dir, 
        mae, 
        r2, 
        model_history_file)


# Funzione per salvare il modello e le metriche
def save_model_and_metrics(model_type: str, model, model_dir, mae, r2, model_history_file):
    # Crea la sottocartella MODELS se non esiste
    models_dir = 'xtream-ai-assignment-developer-main\homework\models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Crea la sottocartella basata sul tipo di modello
    sub_dir = os.path.join(models_dir, f"{model_type}_LM" if model_type == 'linear_regression' else f"{model_type}_XGB")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Genera il timestamp corrente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crea il nome del file per il modello usando il timestamp
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(sub_dir, model_filename)
    
    # Salva il modello nel file system
    joblib.dump(model, model_path)
    
    # Crea un DataFrame con le metriche
    metrics = pd.DataFrame({
        'model_type': [model_type],
        'timestamp': [timestamp],
        'mae': [mae],
        'r2': [r2]
    })

    # Salva le metriche nel file CSV
    if not os.path.exists(model_history_file):
        metrics.to_csv(model_history_file, index=False)
    else:
        metrics.to_csv(model_history_file, mode='a', header=False, index=False)


# Funzione per iperparametri
def objective(
        trial: optuna.trial.Trial,
        X_train_XGB,
        y_train_XGB) -> float:
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
    x_train, x_val, y_train, y_val = train_test_split(X_train_XGB, y_train_XGB, test_size=0.2, random_state=42)

    # Train the model
    model = xgboost.XGBRegressor(**param)
    model.fit(x_train, y_train)

    # Make predictions
    preds = model.predict(x_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, preds)

    return mae





















# Funzione per caricare il modello migliore
# def load_best_model(model_dir, metric='r2', higher_is_better=True):
#     # Crea il percorso del file per le metriche
#     model_history_file = os.path.join(os.path.dirname(__file__), 'model_history.csv')
    
#     # Legge il file CSV contenente le metriche
#     metrics_df = pd.read_csv(model_history_file)
    
#     # Filtra le metriche per tipo di modello
#     lm_metrics_df = metrics_df[metrics_df['model_type'] == 'linear_regression']
#     xgb_metrics_df = metrics_df[metrics_df['model_type'] == 'xgboost']
    
#     # Determina il miglior modello per ogni tipo
#     if higher_is_better:
#         best_lm_metric_idx = lm_metrics_df[metric].idxmax()
#         best_xgb_metric_idx = xgb_metrics_df[metric].idxmax()
#     else:
#         best_lm_metric_idx = lm_metrics_df[metric].idxmin()
#         best_xgb_metric_idx = xgb_metrics_df[metric].idxmin()
    
#     best_lm_metric_row = lm_metrics_df.loc[best_lm_metric_idx]
#     best_xgb_metric_row = xgb_metrics_df.loc[best_xgb_metric_idx]
    
#     # Confronta i migliori modelli e determina il migliore assoluto
#     if higher_is_better:
#         best_metric_row = best_lm_metric_row if best_lm_metric_row[metric] > best_xgb_metric_row[metric] else best_xgb_metric_row
#     else:
#         best_metric_row = best_lm_metric_row if best_lm_metric_row[metric] < best_xgb_metric_row[metric] else best_xgb_metric_row
    
#     best_timestamp = best_metric_row['timestamp']
#     best_model_type = best_metric_row['model_type']
    
#     # Crea il nome della sottocartella basato sul tipo di modello
#     sub_dir = os.path.join(model_dir, 'MODELS', f"{model_type}_LM" if best_model_type == 'linear_regression' else f"{model_type}_XGB")
    
#     # Crea il nome del file per il modello migliore usando il timestamp
#     best_model_filename = f"model_{best_timestamp}.pkl"
#     best_model_path = os.path.join(sub_dir, best_model_filename)
    
#     # Carica il modello migliore
#     best_model = joblib.load(best_model_path)
    
#     # Restituisce il modello migliore e le sue metriche
#     return best_model, best_metric_row.to_dict()
