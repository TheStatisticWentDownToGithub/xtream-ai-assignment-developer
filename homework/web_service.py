from flask import Flask, request, jsonify
import joblib
import pandas as pd
import datetime
import numpy as np
import os
import sqlite3

# Per lettura file
from costants import (
    data_file_path,
    db_path,
    model_dir_service,
)
from functions import (
    load_data,
    data_prep_xgb,
)

# Inizializzazione
app = Flask(__name__)

# Function to initialize the SQLite database
def initialize_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            request_method TEXT,
            request_endpoint TEXT,
            response_body TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database
initialize_db()


# Decoratori
@app.before_request
def log_request():
    if request.endpoint != 'get_logs':
        request.json_data = request.get_json()
    

@app.after_request
def log_response(response):
    if request.endpoint != 'get_logs':
        log_entry = (
            datetime.datetime.utcnow().isoformat(),
            request.method,
            request.endpoint,
            str(response.get_json())
        )
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO api_logs (timestamp, request_method, request_endpoint, response_body)
            VALUES (?, ?, ?, ?)
        ''', log_entry)
        conn.commit()
        conn.close()
    return response

# Carica il modello addestrato
model_trained = os.listdir(model_dir_service)[0]
model_path = os.path.join(model_dir_service, model_trained) 
model = joblib.load(model_path)

# Carica il dataset di addestramento
training_data = load_data(data_file_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid input, no JSON received'}), 400
    
    # Assumi che i dati siano un dizionario con le caratteristiche del diamante
    print("Received data:", data)
    features = pd.DataFrame([data])
    features = data_prep_xgb(features)
    print("DataFrame constructed from received data:\n", features)
    
    # Assicurati che tutte le caratteristiche richieste siano presenti nel DataFrame
    required_features = ['carat', 'cut', 'color', 'clarity']  # Aggiungi altre caratteristiche richieste dal modello
    for feature in required_features:
        if feature not in features.columns:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
    
    prediction = model.predict(features)
    return jsonify({'Predicted diamond value:': prediction.tolist()[0]}), 200
    

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.json
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    weight = data['carat']
    n = data.get('n', 5)  # Numero di campioni simili da restituire, default a 5

    # Filtra il dataset per trovare i diamanti con le stesse caratteristiche
    filtered_data = training_data[
        (training_data['cut'] == cut) &
        (training_data['color'] == color) &
        (training_data['clarity'] == clarity)
    ]

    if filtered_data.empty:
        return jsonify({'error': 'No similar diamonds found.'}), 404
    
    # Trova i `n` diamanti con il peso più simile
    filtered_data['weight_diff'] = np.abs(filtered_data['carat'] - weight)
    similar_diamonds = filtered_data.nsmallest(n, 'weight_diff').drop(columns=['weight_diff'])
    
    return jsonify(similar_diamonds.to_dict(orient='index'))

# Interrogazione DB
@app.route('/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM api_logs')
    logs = cursor.fetchall()
    print(logs, type(logs))
    conn.close()

    return jsonify([{
        'timestamp': log[1],
        'request_method': log[2],
        'request_endpoint': log[3],
        'response_body': log[4]
    } for log in logs])


# Main runner
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)