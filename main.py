from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
from model import predict, train_model
import time

# Configuración del logger
logger.add("api.log", rotation="500 MB", level="INFO")

app = FastAPI(
    title="API de Predicción de Calidad del Vino",
    description="API REST para predecir la calidad del vino basado en sus características químicas",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: List[float] = None

    class Config:
        schema_extra = {
            "example": {
                "features": [
                    13.2,  # alcohol (% vol)
                    1.78,  # malic_acid (g/L)
                    2.14,  # ash (g/L)
                    11.2,  # alcalinity_of_ash (g/L)
                    100,   # magnesium (mg/L)
                    2.4,   # total_phenols (g/L)
                    2.4,   # flavanoids (g/L)
                    0.23,  # nonflavanoid_phenols (g/L)
                    1.71,  # proanthocyanins (g/L)
                    3.9,   # color_intensity (absorbancia)
                    1.3,   # hue (absorbancia)
                    2.5,   # OD280/OD315 of diluted wines
                    1100   # proline (mg/L)
                ]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_time: float

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando el servicio de API")
    # Entrenamos el modelo al iniciar el servicio
    train_model()
    logger.info("Modelo entrenado y listo para usar")

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de Predicción de Calidad del Vino",
        "endpoints": {
            "GET /": "Muestra esta información",
            "POST /predict": "Realiza una predicción de la calidad del vino. Requiere un JSON con las características del vino",
            "GET /health": "Verifica el estado de la API"
        },
        "ejemplo_predict": {
            "método": "POST",
            "url": "/predict",
            "body": {
                "features": [
                    13.2,  # alcohol (% vol)
                    1.78,  # malic_acid (g/L)
                    2.14,  # ash (g/L)
                    11.2,  # alcalinity_of_ash (g/L)
                    100,   # magnesium (mg/L)
                    2.4,   # total_phenols (g/L)
                    2.4,   # flavanoids (g/L)
                    0.23,  # nonflavanoid_phenols (g/L)
                    1.71,  # proanthocyanins (g/L)
                    3.9,   # color_intensity (absorbancia)
                    1.3,   # hue (absorbancia)
                    2.5,   # OD280/OD315 of diluted wines
                    1100   # proline (mg/L)
                ]
            }
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        start_time = time.time()
        
        # Validamos que tengamos el número correcto de características
        if len(request.features) != 13:
            raise HTTPException(
                status_code=400,
                detail="Se requieren exactamente 13 características químicas del vino para la predicción"
            )
        
        # Realizamos la predicción
        result = predict(request.features)
        
        # Calculamos el tiempo de predicción
        prediction_time = time.time() - start_time
        
        # Registramos la predicción exitosa
        logger.info(
            f"Predicción exitosa - Clase: {result['prediction']}, "
            f"Probabilidad: {result['probability']:.4f}, "
            f"Tiempo: {prediction_time:.4f}s"
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            prediction_time=prediction_time
        )
        
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}