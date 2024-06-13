import os
from sklearn.model_selection import train_test_split

# Funzioni utilizzate
from functions import (
    load_data,
    data_prep_lm,
    data_prep_xgb,
    train_and_evaluate_model,
)

# Import dei percorsi
from costants import (
    data_file_path,
    model_history_file,
    model_dir
)

# Assicurarsi che la directory per i modelli esista
os.makedirs(model_dir, exist_ok=True)

# Caricare i dati freschi originali
diamonds = load_data(data_file_path, nrows=10)  # imposta nrows:int per scegliere se prendere un sottoinsieme del df

# genero due dataframe sistemati: Uno per gli LM ed uno per gli XGB
diamonds_LM = data_prep_lm(
    df=diamonds,
    drop_cols = ['depth', 'table', 'y', 'z'],
    to_dummies = ['cut', 'color', 'clarity'],
    to_log = None
)
diamonds_XGB = data_prep_xgb(diamonds)

# Definizione target column
target_column = 'price'  # Sostituisci con il nome della colonna target

# generazione Train e Test Per LM
y_LM = diamonds_LM[target_column]
X_LM = diamonds_LM.drop(columns=[target_column])
X_train_LM, X_test_LM, y_train_LM, y_test_LM = train_test_split(X_LM, y_LM, test_size=0.2, random_state=42)

# generazione Train e Test Per XGB
y_XGB = diamonds_XGB[target_column]
X_XGB = diamonds_XGB.drop(columns=[target_column])
X_train_XGB, X_test_XGB, y_train_XGB, y_test_XGB = train_test_split(X_XGB, y_XGB, test_size=0.2, random_state=42)

#    carat    cut color clarity  depth  table     x     y     z
# 0    1.1  Ideal     H     SI2   62.0   55.0  6.61  6.65  4.11


# Addestramento LM (model_name = 'linear_regression)
train_and_evaluate_model(
    model_name = 'linear_regression', 
    X_train = X_train_LM, 
    X_test = X_test_LM, 
    y_train = y_train_LM, 
    y_test = y_test_LM,  
    model_dir = model_dir,
    model_history_file = model_history_file,
    target_log_LM = True,
    grid_search_XGB = None)

# Addestramento LM con target log (model_name = 'log_linear_regression)
train_and_evaluate_model(
    model_name = 'linear_regression', 
    X_train = X_train_LM, 
    X_test = X_test_LM, 
    y_train = y_train_LM, 
    y_test = y_test_LM,  
    model_dir = model_dir,
    model_history_file = model_history_file,
    target_log_LM = False,
    grid_search_XGB = None)

# Addestramento XGB(model_name = 'xgboost)
train_and_evaluate_model(
    model_name = 'xgboost', 
    X_train = X_train_XGB, 
    X_test = X_test_XGB, 
    y_train = y_train_XGB, 
    y_test = y_test_XGB,  
    model_dir = model_dir,
    model_history_file = model_history_file,
    target_log_LM = False,
    grid_search_XGB = False)

# Addestramento XGB con ottimizzazione(model_name = 'opt_xgboost)
train_and_evaluate_model(
    model_name = 'xgboost', 
    X_train = X_train_XGB, 
    X_test = X_test_XGB, 
    y_train = y_train_XGB, 
    y_test = y_test_XGB,  
    model_dir = model_dir,
    model_history_file = model_history_file,
    target_log_LM = False,
    grid_search_XGB = True)

print(f"All models outputs produced..")


