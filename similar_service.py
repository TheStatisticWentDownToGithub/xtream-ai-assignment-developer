import requests as rq
import pandas as pd

### SIMILAR ###
# Record to pass 
data = {
    'carat': [1.1],
    'cut': ['Ideal'],
    'color': ['H'],
    'clarity': ['SI2'],
    'depth': [62.0],
    'table': [55.0],
    'price': [4733],
    'x': [6.61],
    'y': [6.65],
    'z': [4.11]
}
data_try = pd.DataFrame(data)

print('Request similar rows of ...')
print(data_try)
print('\n')
# Converti il DataFrame in un dizionario
data_dict = data_try.to_dict(orient='records')[0]

# URL del server
path = 'http://127.0.0.1:5000'

# Invia la richiesta POST alla tua API
response = rq.post(path + '/find_similar', json=data_dict)
print(response.json())