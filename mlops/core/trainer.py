import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from trainer.utils.ZeroDCE import ZeroDCE

IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400

#Open Cloud Bucket
BUCKET = 'gs://zero-dce/'

#Data Preprocessing
def load_data(image_path):
    #Load the image file
    #decode the png
    image = tf.image.decode_png(image, channels=3)
    #make all images 256*256 
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    #Normalize it image = tf.io.read_file(image_path)
    image = image / 255.0
    return image

#Generate tensor slices from the dataset and create training batches 
def data_generator(low_light_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

#Train data 
def train():
    #Split dataset into training, validation and test datasets
    train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
    val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
    test_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))

    train_dataset = data_generator(train_low_light_images)
    val_dataset = data_generator(val_low_light_images)

    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)

	#Run model training 
    zero_dce_model = ZeroDCE()
    zero_dce_model.compile(learning_rate=1e-4)
    history = zero_dce_model.fit(train_dataset, validation_data=val_dataset, epochs=100)

	#Save model and weights 
    zero_dce_model.dce_model.save(BUCKET + 'model/')
    zero_dce_model.save_weights(BUCKET + 'weights/')