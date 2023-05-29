# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:04:17 2023

@author: bahman ravaei
"""
# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


# Define some parameters for the loader
batch_size = 32
img_height = 64
img_width = 64
dataset_dir="flower_photos"


# Load dataset
train_ds,test_ds = image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="both",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""
train_ds = image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
"""


# Part 1 - Data Preprocessing

# Standardize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))



# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=len(train_ds.class_names), activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
#cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train_ds, validation_data = test_ds, epochs = 25)

# Part 4 - Making a single prediction

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

test_image_list=['testpic/pic1.jpg', 'testpic/pic2.jpg', 'testpic/pic3.jpg']
#plt.figure(figsize=(9, 3))
fig, axs = plt.subplots(3)
index=0
for image_path in test_image_list:
    test_image = tf.keras.utils.load_img(image_path, target_size=(64,64))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.array([test_image])  # Convert single image to a batch.
    result = cnn.predict(test_image)
    axs[index].bar(train_ds.class_names, result.tolist()[0])
    #plt.bar(train_ds.class_names, result.tolist()[0])
    index+=1
