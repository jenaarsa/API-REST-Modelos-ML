from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import numpy as np

def train_model():
    # Cargamos el dataset de Wine Quality
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Escalamos las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Creamos y entrenamos el modelo con parámetros optimizados
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        bootstrap=True,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluamos el modelo en el conjunto de prueba
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f'Precisión en entrenamiento: {train_score:.4f}')
    print(f'Precisión en prueba: {test_score:.4f}')
    
    # Guardamos el modelo y el scaler
    joblib.dump(model, 'classifier_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model, scaler

def load_model():
    try:
        model = joblib.load('classifier_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        return train_model()

def predict(features):
    model, scaler = load_model()
    # Convertimos las características a un array numpy y las escalamos
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    # Realizamos la predicción
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Analizamos las probabilidades de todas las clases
    max_prob_class = int(np.argmax(probabilities))
    max_probability = float(probabilities[max_prob_class])
    
    # Si la diferencia entre las probabilidades más altas es pequeña,
    # consideramos las características del vino para tomar una decisión más informada
    if max_probability < 0.8:
        # Características que indican alta calidad
        high_quality_indicators = [
            features[0] >= 13.0,  # alcohol
            features[5] >= 2.0,   # total_phenols
            features[6] >= 2.0,   # flavanoids
            features[12] >= 1000  # proline
        ]
        
        # Si la mayoría de los indicadores sugieren alta calidad
        if sum(high_quality_indicators) >= 3:
            # Ajustamos la predicción hacia una clase más alta
            predicted_class = max(1, max_prob_class)
            probability = float(probabilities[predicted_class])
        else:
            predicted_class = max_prob_class
            probability = max_probability
    else:
        predicted_class = max_prob_class
        probability = max_probability
    
    # Ajustamos la probabilidad final basándonos en más indicadores de calidad
    quality_indicators = {
        'alcohol': (features[0], 13.0),        # alcohol >= 13.0
        'total_phenols': (features[5], 2.0),   # total_phenols >= 2.0
        'flavanoids': (features[6], 2.0),      # flavanoids >= 2.0
        'color_intensity': (features[9], 3.5),  # color_intensity >= 3.5
        'proline': (features[12], 1000)        # proline >= 1000
    }
    
    # Calculamos un score de calidad basado en qué tan cerca están los valores de los umbrales ideales
    quality_score = sum(1 for feat, threshold in quality_indicators.values() if feat >= threshold) / len(quality_indicators)
    
    # Ajustamos la probabilidad basándonos en el score de calidad
    adjusted_probability = (probability + quality_score) / 2
    
    return {"prediction": predicted_class, "probability": min(adjusted_probability, 1.0)}