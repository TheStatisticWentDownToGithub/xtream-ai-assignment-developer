import requests as rq

# URL del server
path = 'http://127.0.0.1:5000'

# Invia la richiesta GET alla tua API
response = rq.get(path + '/logs')
print(response.text)