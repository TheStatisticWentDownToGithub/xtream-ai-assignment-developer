import requests as rq
# Per lettura file
from functions import (
    load_data,
)

### SIMILAR ###
# Record to pass
data_try = load_data(r'C:\Users\daniele\Desktop\esami colloqui\Xtreme\xtream-ai-assignment-developer-main\data\diamonds.csv', nrows=1)

# Converti il DataFrame in un dizionario
data_dict = data_try.to_dict(orient='records')[0]

# URL del server
path = 'http://127.0.0.1:5000'

# Invia la richiesta POST alla tua API
response = rq.post(path + '/find_similar', json=data_dict)
print(response.json())