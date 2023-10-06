
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
import numpy as np
import os
from utils import *
import pathlib
import pandas as pd

MODEL_PATH = MODELS_DIR.joinpath('models', 'final_nn')
model = tf.keras.models.load_model(MODEL_PATH)

# Normalize data here
training_data_path = DATA_DIR.joinpath('reference','california_data.csv')
raw_df = pd.read_csv(training_data_path)
# Although data are mostly pre-processed, drop nan and convert to binary for classifier
cleaned_df = raw_df.dropna()
# Reclassify yolo training data to binary
cleaned_df = cleaned_df.replace({'training_class': {10: 0, 11:0, 12:0, 13:0}})
cleaned_df = cleaned_df.drop('date', axis=1)
# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)
# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('training_class'))
val_labels = np.array(val_df.pop('training_class'))
test_labels = np.array(test_df.pop('training_class'))

variables = ['cor', 'pha', 'dif', 'ref', 'spw', 'vel']

train_features = np.array(train_df[variables])
val_features = np.array(val_df[variables])
test_features = np.array(test_df[variables])
# normalize data instead of standardize
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_features, batch_size=32)

classdir = pathlib.Path('/Volumes/backupdata/doppler/data/classified')

def classify(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.tif'):
                # print(file)
                classified_file = 'classified_' + file
                image_file = (os.path.join(classdir, classified_file))
                # print(image_file)
                if os.path.exists(image_file) == False:
                    image_file2 = (os.path.join(root, file))
                    # print('image does not exist')
                    classify_image(image_file2, file, model, normalizer, classdir)