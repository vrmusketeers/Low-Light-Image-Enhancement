import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from trainer.utils.LossFunctions import *

"""
Extend the Keras.Model to add the loss functions.
ZeroDCE utilizes Adam optimizer 8 convolutional layers and
5 convolutional layers set to ReLU activation 
"""
class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = self.build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
    
    
    def build_dce_net(self):
        input_img = keras.Input(shape=[None, None, 3])
        conv1 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(input_img)
        conv2 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(conv1)
        conv3 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(conv2)
        conv4 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(conv3)
        int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
        conv5 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(int_con1)
        int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
        conv6 = layers.Conv2D(
            32, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(int_con2)
        int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
        x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
            int_con3
        )
        return keras.Model(inputs=input_img, outputs=x_r)