import requests as rq
# Per lettura file
from functions import (
    load_data,
)

### PREDICTION ###
# Data su cui fare prediction
data_try = load_data(r'C:\Users\daniele\Desktop\esami colloqui\Xtreme\xtream-ai-assignment-developer-main\data\diamonds.csv', nrows=1)
data_try.drop(columns='price', inplace=True)

# Converti il DataFrame in un dizionario
data_dict = data_try.to_dict(orient='records')[0]

# URL del server
path = 'http://127.0.0.1:5000'

# Invia la richiesta POST alla tua API
response = rq.post(path + '/predict', json=data_dict)
print(response.json())