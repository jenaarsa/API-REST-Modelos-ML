# API REST para Modelo de Clasificación

Este proyecto implementa una API REST utilizando FastAPI para servir un modelo de clasificación. El servicio incluye logging, monitoreo y documentación automática de endpoints.

## Requisitos

Asegúrate de tener Python 3.8 o superior instalado. Los requisitos se encuentran en el archivo `requirements.txt`.

## Instalación

1. Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
.\venv\Scripts\activate  # En Windows
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución del Servicio

Para iniciar el servidor:
```bash
uvicorn main:app --reload
```

El servidor estará disponible en `http://localhost:8000`

## Endpoints Disponibles

### GET /
- Endpoint de bienvenida
- Retorna un mensaje de bienvenida

### GET /health
- Endpoint de verificación de salud del servicio
- Retorna el estado actual del servicio y timestamp

### POST /predict
- Endpoint para realizar predicciones
- Requiere un JSON con el siguiente formato:
```json
{
    "features": [0.1, 0.2, 0.3, 0.4]
}
```
- Retorna la predicción, probabilidad y tiempo de procesamiento

## Documentación de la API

FastAPI genera automáticamente la documentación de la API. Puedes acceder a:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Monitoreo y Logging

El servicio utiliza Loguru para el registro de eventos. Los logs se guardan en el archivo `api.log` y incluyen:
- Inicio del servicio
- Predicciones exitosas
- Errores y excepciones

## Ejemplo de Uso con curl

```bash
# Verificar el estado del servicio
curl http://localhost:8000/health

# Realizar una predicción
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [14.2, 2.4, 2.5, 20, 92, 2.9, 2.9, 0.32, 1.8, 4.8, 1.1, 3.85, 1250]}'
```

## Notas Adicionales

- El modelo se entrena automáticamente al iniciar el servicio
- Se incluye validación de datos para asegurar el formato correcto de las entradas
- El sistema de logging registra todas las predicciones y posibles errores