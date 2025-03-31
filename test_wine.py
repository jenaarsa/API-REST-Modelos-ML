import requests

# Diferentes conjuntos de parámetros para probar
wine_samples = [
    # Vino de alta calidad
    [14.2, 2.4, 2.5, 20, 92, 2.9, 2.9, 0.32, 1.8, 4.8, 1.1, 3.85, 1250],
    # Vino de calidad media
    [12.5, 2.0, 2.2, 18, 85, 2.2, 2.2, 0.28, 1.5, 3.9, 1.0, 3.3, 1000],
    # Vino de baja calidad
    [11.0, 1.5, 1.8, 15, 75, 1.8, 1.5, 0.25, 1.2, 3.0, 0.9, 2.9, 800]
]

# URL del endpoint
url = 'http://127.0.0.1:8000/predict'

# Probar cada conjunto de parámetros
for i, features in enumerate(wine_samples, 1):
    try:
        response = requests.post(url, json={'features': features})
        result = response.json()
        print(f'\nMuestra {i}:')
        print(f'Características: {features}')
        print(f'Predicción: {result}')
    except Exception as e:
        print(f'Error con la muestra {i}: {e}')