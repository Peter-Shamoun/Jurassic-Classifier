import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import io
from PIL import Image
def model_prediction(image):
    def binary_to_matrix(binary_data):
        with io.BytesIO(binary_data) as image_file:

            image = Image.open(image_file)
            image_matrix = np.array(image)
        return image_matrix

    image = binary_to_matrix(image)
    
    
    def resize_image(image):
        target_size = (256, 256)
        image = Image.fromarray(image)
        resized_image = image.resize((256, 256), Image.Resampling.LANCZOS)

        resized_matrix = np.array(resized_image)
        return resized_matrix
    
    image = resize_image(image) 
    
    
    
    image = image / 255
    
    dino_to_int = {'Tyrannosaurus_Rex': 0,
     'Gallimimus': 1,
     'Parasaurolophus': 2,
     'Microceratus': 3,
     'Spinosaurus': 4,
     'Triceratops': 5,
     'Velociraptor': 6,
     'Pachycephalosaurus': 7,
     'Dilophosaurus': 8,
     'Stegosaurus': 9,
     'Compsognathus': 10,
     'Ankylosaurus': 11,
     'Dimorphodon': 12,
     'Corythosaurus': 13,
     'Brachiosaurus': 14}
    
    ensemble_models = []
    for i in range(3):
        model_path = f"./ensemble_model_{i}.h5"
        ensemble_models.append(load_model(model_path))
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(image):
        sample = np.expand_dims(image, axis=0)


        predictions = [model.predict(sample)[0] for model in ensemble_models]


        average_predictions = np.mean(predictions, axis=0)


        average_probabilities = softmax(average_predictions)


        predicted_index = np.argmax(average_probabilities)


        prediction = {v: k for k, v in dino_to_int.items()}[predicted_index]


        return prediction
    image = predict(image)
    return image
