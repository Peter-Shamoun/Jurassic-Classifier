import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import io
from PIL import Image
import time
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Model metadata
MODEL_VERSION = "1.0.0"
MODEL_CLASSES = [
    "Tyrannosaurus_Rex", "Gallimimus", "Parasaurolophus", 
    "Microceratus", "Spinosaurus", "Triceratops", 
    "Velociraptor", "Pachycephalosaurus", "Dilophosaurus", 
    "Stegosaurus", "Compsognathus", "Ankylosaurus", 
    "Dimorphodon", "Corythosaurus", "Brachiosaurus"
]
IMAGE_SIZE = (256, 256)
MAX_IMAGE_DIMENSION = 4000  # Maximum allowed image dimension

# Global model cache
_model_cache = {}

def get_model_info():
    """Return model version and metadata"""
    return {
        "version": MODEL_VERSION,
        "classes": len(MODEL_CLASSES),
        "class_names": [name.replace("_", " ") for name in MODEL_CLASSES],
        "input_shape": (*IMAGE_SIZE, 3),
        "ensemble_size": 3,
        "last_loaded": _model_cache.get("last_loaded", None)
    }

def load_ensemble_models():
    """Load models only once and cache them"""
    global _model_cache
    if "ensemble" not in _model_cache:
        logger.info("Loading ensemble models...")
        start_time = time.time()
        models = []
        for i in range(3):
            model_path = f"ensemble_model_{i}.h5"
            if os.path.exists(model_path):
                logger.info(f"Loading model {i} from {model_path}")
                models.append(load_model(model_path))
            else:
                error_msg = f"Model file {model_path} not found"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        _model_cache["ensemble"] = models
        _model_cache["last_loaded"] = time.strftime("%Y-%m-%d %H:%M:%S")
        loading_time = time.time() - start_time
        logger.info(f"All models loaded in {loading_time:.2f}s")
    
    return _model_cache["ensemble"]

def binary_to_matrix(binary_data):
    """Convert binary image data to a numpy matrix"""
    try:
        with io.BytesIO(binary_data) as image_file:
            image = Image.open(image_file)
            image_matrix = np.array(image)
            
            # Validate image dimensions
            height, width = image_matrix.shape[:2]
            if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
                raise ValueError(f"Image dimensions ({width}x{height}) exceed maximum allowed size ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")
                
            return image_matrix
    except Exception as e:
        logger.error(f"Error converting binary to image: {str(e)}")
        raise

def resize_image(image):
    """Resize image to target dimensions"""
    try:
        image = Image.fromarray(image)
        resized_image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        resized_matrix = np.array(resized_image)
        return resized_matrix
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise

def softmax(x):
    """Apply softmax function to convert logits to probabilities"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict(image):
    """Run prediction with ensemble models"""
    # Ensure models are loaded
    ensemble_models = load_ensemble_models()
    
    # Prepare input
    sample = np.expand_dims(image, axis=0)
    
    # Get predictions from each model
    predictions = []
    for i, model in enumerate(ensemble_models):
        start_time = time.time()
        pred = model.predict(sample, verbose=0)[0]
        predictions.append(pred)
        logger.debug(f"Model {i} prediction time: {time.time() - start_time:.4f}s")
    
    # Average predictions and apply softmax
    average_predictions = np.mean(predictions, axis=0)
    probabilities = softmax(average_predictions)
    
    # Get top class and confidence
    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index])
    
    # Get sorted indices for top predictions
    top_indices = np.argsort(probabilities)[::-1]
    
    return {
        "predicted_index": int(predicted_index),
        "top_indices": top_indices.tolist(),
        "probabilities": probabilities.tolist(),
        "predicted_class": MODEL_CLASSES[predicted_index],
        "confidence": confidence
    }

def model_prediction(image):
    """Main prediction function that processes input image and returns prediction"""
    start_time = time.time()
    logger.info("Starting prediction process")
    
    # Process image
    try:
        # Convert binary to matrix
        logger.debug("Converting binary to image matrix")
        image_matrix = binary_to_matrix(image)
        
        # Resize image
        logger.debug("Resizing image")
        resized_image = resize_image(image_matrix)
        
        # Normalize pixel values
        logger.debug("Normalizing image")
        normalized_image = resized_image / 255.0
        
        # Get prediction
        logger.debug("Running prediction")
        prediction_result = predict(normalized_image)
        
        # Format result for API response
        class_name = prediction_result["predicted_class"]
        confidence = prediction_result["confidence"]
        probabilities = prediction_result["probabilities"]
        
        # Get top 3 predictions
        top_predictions = []
        for idx in prediction_result["top_indices"][:3]:
            top_predictions.append({
                "species": MODEL_CLASSES[idx].replace("_", " "),
                "confidence": float(probabilities[idx])
            })
        
        result = {
            "prediction": class_name.replace("_", " "),
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model_version": MODEL_VERSION,
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Prediction complete: {class_name} ({confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise
