
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *
import pandas as pd
from tensorflow import keras
import numpy as np
import os
import pathlib

MODEL_PATH = MODELS_DIR.joinpath('final_nn')
model = tf.keras.models.load_model(MODEL_PATH)

def normal_data(training_df):
    # Normalize data here

    raw_df = pd.read_csv(training_df)
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
    return normalizer

training_data_path = DATA_DIR.joinpath('reference', 'california_data.csv')
normalizer = normal_data(training_data_path)


def classify(rootdir, classify_dir):
    for root, dirs, files in os.walk(rootdir):
        if '2_scan_agg' in root:
            for file in files:
                if file.endswith('.tif'):
                    classified_file = 'classified_' + file
                    image_file = (os.path.join(classify_dir, classified_file))
                    if os.path.exists(image_file) == False:
                        image_file2 = (os.path.join(root, file))
                        # print('image does not exist')
                        classify_image(image_file2, file, model, normalizer, classify_dir)