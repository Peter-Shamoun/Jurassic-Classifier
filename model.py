import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Flatten, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Maximum, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Add this before reading the pickle file
if os.path.exists("dino_pics.pkl"):
    print(f"File size: {os.path.getsize('dino_pics.pkl')} bytes")
    data = pd.read_pickle("dino_pics.pkl")
else:
    raise FileNotFoundError("dino_pics.pkl not found in current directory")

data = data[data['resized_image_matrix'].apply(
    lambda x: x.shape == (256, 256, 3))] #filtering out ~5 row with wrong shpe



dinos = data["dino_name"].unique()
dino_to_int = {dinos[i]: i for i in range(len(dinos))}


data['ints'] = data["dino_name"].apply(lambda x: dino_to_int[x])



data['ints'] = data["dino_name"].apply(lambda x: dino_to_int[x])

data['resized_image_matrix'] = data['resized_image_matrix'].apply(lambda x:x/255) #normalizing entries

X = data['resized_image_matrix']
y = data['ints'] #X and y

X = X.to_numpy() # series to numpy array
y = y.to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #train-test split

for i in range(len(X_train)):
    X_train[i] = X_train[i].astype('float32')
for i in range(len(X_val)):
    X_val[i] = X_val[i].astype('float32') #make them the same dtype

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

X_train_stacked = np.stack(X_train).astype('float32')
X_val_stacked = np.stack(X_val).astype('float32')


y_train_one_hot = np.stack(y_train).astype('float32')
y_val_one_hot = np.stack(y_val).astype('float32')
#stack the arrays to make them 4D arrays to prepare for CNN

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(11,11), strides=(8,8), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))  
    return model

# list of models for the ensemble
ensemble_models = [create_model((256, 256, 3)) for _ in range(3)]

# Compile the models
for model in ensemble_models:
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

# Fit the models
for model in ensemble_models:
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])


for model in ensemble_models:
    model.fit(X_train_stacked, y_train_one_hot, batch_size=32, epochs=100, validation_data=(X_val_stacked, y_val_one_hot))


sample = np.expand_dims(X[-1], axis=0)

# Predict with each model in the ensemble and store the predictions
predictions = [model.predict(sample)[0] for model in ensemble_models]

# Average the predictions across all models
average_predictions = np.mean(predictions, axis=0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
# Apply softmax to convert logits to probabilities
average_probabilities = softmax(average_predictions)

#index of the highest probability
predicted_index = np.argmax(average_probabilities)

#class name
prediction = {v: k for k, v in dino_to_int.items()}[predicted_index]

for i, model in enumerate(ensemble_models):
    model_path = f"./ensemble_model_{i}.h5"  # .h5 extension for HDF5 format
    model.save(model_path)
    print(f"Model {i} saved to {model_path}")
