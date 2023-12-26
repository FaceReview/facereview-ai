# -*- coding: cp949 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import io
import os
import cv2
import math
import glob
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import time
import copy


def bottleneck_residual_block(X, filters, reduce=False, s=2):
    F1, F2, F3 = filters
    X_shortcut = X
    
    if reduce:
        X = keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.ReLU()(X)
        
        X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)
    else: 
        X = keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.ReLU()(X)
    
    X = keras.layers.Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.ReLU()(X)
    
    return X


def ResNet50(classes):
    X_input = keras.layers.Input(shape=[96, 96, 1])

    X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_input)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = bottleneck_residual_block(X, [64, 64, 256], reduce=True, s=1)
    X = bottleneck_residual_block(X, [64, 64, 256])
    X = bottleneck_residual_block(X, [64, 64, 256])

    X = bottleneck_residual_block(X, [128, 128, 512], reduce=True)
    X = bottleneck_residual_block(X, [128, 128, 512])
    X = bottleneck_residual_block(X, [128, 128, 512])
    X = bottleneck_residual_block(X, [128, 128, 512])

    X = bottleneck_residual_block(X, [256, 256, 1024], reduce=True)
    X = bottleneck_residual_block(X, [256, 256, 1024])
    X = bottleneck_residual_block(X, [256, 256, 1024])
    X = bottleneck_residual_block(X, [256, 256, 1024])
    X = bottleneck_residual_block(X, [256, 256, 1024])
    X = bottleneck_residual_block(X, [256, 256, 1024])

    X = bottleneck_residual_block(X, [512, 512, 2048], reduce=True)
    X = bottleneck_residual_block(X, [512, 512, 2048])
    X = bottleneck_residual_block(X, [512, 512, 2048])

    X = keras.layers.AveragePooling2D((1,1))(X)

    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(units=512, activation='relu')(X)
    X = keras.layers.Dense(units=classes, activation='softmax')(X)
    
    model = keras.models.Model(inputs=X_input, outputs=X)

    return model

df = pd.read_csv('./data/dataset.csv')

xs = np.array(df['path'])
ys = np.array(df['label'])

train_x, valid_x, train_y, valid_y = train_test_split(xs, ys, train_size=0.7)
# valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5)

train_x = np.array([cv2.imread(item, cv2.IMREAD_GRAYSCALE) / 255 for item in train_x]).reshape(train_x.shape[0], 96, 96, 1)
valid_x = np.array([cv2.imread(item, cv2.IMREAD_GRAYSCALE) / 255 for item in valid_x]).reshape(valid_x.shape[0], 96, 96, 1)
# test_x = np.array([cv2.imread(item, cv2.IMREAD_GRAYSCALE) / 255 for item in test_x]).reshape(test_x.shape[0], 96, 96, 1)

train_y = tf.keras.utils.to_categorical(train_y, 5)
valid_y = tf.keras.utils.to_categorical(valid_y, 5)
# test_y = tf.keras.utils.to_categorical(test_y, 5)

#
#
#

model = ResNet50(5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

#
#
#

start_time = time.time()

# callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='model1000.h5', monitor='val_loss', save_best_only=True)
# callbacks = EarlyStopping(patience = 10)

callbacks = [
  EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.0001, patience = 20),
  ModelCheckpoint(filepath='data/model.h5', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    train_x, train_y, validation_data=(valid_x, valid_y),
    epochs=2000, batch_size=16, callbacks=[callbacks]
)
# score = model.evaluate(test_x, test_y)
# print("accuracy : ",score[1],"    loss : ",score[0])

print(f'{time.time() - start_time}초 동안 학습함.')

with open('data/model.txt', 'w', encoding='UTF-8') as f:
  for key, value in history.history.items():
    f.write(f'{key} : {value}\n')



