import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Reshape,Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint    
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint    
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.losses import mean_squared_logarithmic_error
from tensorflow.keras.optimizers import Adam
import time
import pickle
import chromadb
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import sqlite3
from sklearn.preprocessing import LabelEncoder


def extract_dino_info(filename):
    '''
    Function that extracts the dinosoaur's info give nthe filename
    '''
    split_filename = filename.split('/')
    dino_name = split_filename[5]
    dino_number = split_filename[-1].split('_')[1].split('.')[0]
    return [dino_name, dino_number]


folder_path = '/dino_pics'

folder_paths = []


for item in os.listdir(folder_path):
    
    full_path = os.path.join(folder_path, item)
    
    if os.path.isdir(full_path) and item != '.DS_Store':
        folder_paths.append(full_path)
conn = sqlite3.connect('dino_images_real.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS data (
        dino_name TEXT,
        dino_number INTEGER,
        dino_pic BLOB
    )
''')

folder_path = '/Users/peter_parker/Downloads/dinosaur_dataset'

folder_paths = []


for item in os.listdir(folder_path):
    
    full_path = os.path.join(folder_path, item)
    
    if os.path.isdir(full_path) and item != '.DS_Store':
        folder_paths.append(full_path)

for folder in folder_paths:
    file_names = [f for f in os.listdir(folder)]
    for filename in file_names:
        with open(folder + '/' + filename, 'rb') as f:
            image = f.read()
        name, number = extract_dino_info(folder+'/'+filename)
        cursor.execute("INSERT INTO data (dino_name, dino_number, dino_pic) VALUES (?, ?, ?)", 
                       (name, number,sqlite3.Binary(image),))

conn.commit()
conn.close()

conn = sqlite3.connect('dino_images_real.db')
chunk_size = 10000
chunks = []
for chunk in pd.read_sql_query("SELECT * FROM data", conn, chunksize=chunk_size):
    chunks.append(chunk)
conn.close()
df = pd.concat(chunks, ignore_index=True)


def binary_to_matrix(binary_data):
    with io.BytesIO(binary_data) as image_file:
        
        image = Image.open(image_file)
        
        image_matrix = np.array(image)
    return image_matrix


df['matrices'] = df['dino_pic'].apply(lambda x: binary_to_matrix(x))

pictures = df.drop(columns = ['dino_pic'])

pictures.drop(columns = ['dino_number'])


target_size = (256, 256)

def resize_image(matrix, size):

    image = Image.fromarray(matrix)

    resized_image = image.resize(size, Image.ANTIALIAS)

    resized_matrix = np.array(resized_image)
    return resized_matrix

# Apply the resizing function to each row in the dataframe
pictures['resized_image_matrix'] = pictures['matrices'].apply(lambda x: resize_image(x, target_size))


label_encoder = LabelEncoder()


pictures['dino_label'] = label_encoder.fit_transform(pictures['dino_name'])

one_hot_encoded = to_categorical(pictures['dino_label'])
one_hot_df = pd.DataFrame(one_hot_encoded, dtype=int)
one_hot_df.columns = label_encoder.classes_


pictures = pd.concat([pictures, one_hot_df], axis=1)

def one_hot_encoding(row):
    one_hot_list = []
    for column in row.index:
        if column not in ['dino_number', 'matrices', 'dino_label', 'resized_image_matrix', 'dino_name', 'dino_one_hot']:t
            one_hot_list.append(row[column])
    return one_hot_list
pictures['one_hot'] = pictures.apply(one_hot_encoding, axis=1)


pictures = pictures.drop(columns = ['dino_number','matrices','dino_label','Ankylosaurus',
                                    'Brachiosaurus','Compsognathus', 'Dilophosaurus','Gallimimus',
                                    'Microceratus', 'Pachycephalosaurus', 'Spinosaurus', 'Stegosaurus',
                                    'Triceratops', 'Tyrannosaurus_Rex', 'Velociraptor'])

pictures = pictures.drop(columns = ['Corythosaurus', 'Dimorphodon', 'Parasaurolophus'])

pictures=pictures.drop(columns = ['dino_one_hot'])
pictures.to_pickle('dino_pics.pkl')
