import numpy as np
import tensorflow as tf
from tensorflow import keras


def dense_model():
    

    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10000, activation='sigmoid'),
        keras.layers.Reshape((100, 100))
    ])

    return model

def dense_model2():
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(8000, activation='relu'),
        keras.layers.Dense(4000, activation='relu'),
        keras.layers.Dense(2000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(2000, activation='relu'),
        keras.layers.Dense(4000, activation='relu'),
        keras.layers.Dense(8000, activation='relu'),
        keras.layers.Dense(10000, activation='sigmoid'),
        keras.layers.Reshape((100, 100))
    ])
    return model

def conv_and_upsampling_model():

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', input_shape=(100,100,1)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding='same'))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding='same'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), padding='same'))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.UpSampling2D(size=(2,2)))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.UpSampling2D(size=(2,2)))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(keras.layers.UpSampling2D(size=(2,2)))

    # model.add(keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='relu'))
    model.add(keras.layers.Conv2D(filters=1, kernel_size=(5,5), activation='sigmoid'))

    return model

