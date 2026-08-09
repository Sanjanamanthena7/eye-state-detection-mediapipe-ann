from pathlib import Path
import pickle

import cv2
import numpy as np
import tensorflow as tf


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "eye_state_ann.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# ==========================================================
# PREPROCESS EYE IMAGE
# ==========================================================

def preprocess_eye(image):

    if image is None:
        raise ValueError("Image could not be loaded.")

    # BGR → grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Same size used for training
    gray = cv2.resize(
        gray,
        (16, 16),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    gray = gray.astype(np.float32) / 255.0

    # 16 × 16 = 256 features
    features = gray.reshape(1, 256)

    return features


# ==========================================================
# PREDICT
# ==========================================================

def predict_eye_state(image):

    features = preprocess_eye(image)

    scaled = scaler.transform(features).astype(np.float32)

    prediction = model.predict(
        scaled,
        verbose=0
    )

    probability = float(prediction[0][0])

    # Binary classifier
    predicted_index = int(probability >= 0.5)

    label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = (
        probability
        if predicted_index == 1
        else 1.0 - probability
    )

    return {
        "label": str(label),
        "confidence": float(confidence),
        "probability_open": float(probability),
        "probability_closed": float(1.0 - probability)
    }