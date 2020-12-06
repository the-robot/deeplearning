import numpy as np

from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def UNet(input_shape: typing.Tuple[int], start_neurons = 64, name = "U-Net") -> Model:
    """
    Implementation of the UNet architecture.

    Arguments:
    input_shape   -- shape of the images of the dataset
    start_neuron  -- defines the smallest filter
                     in UNet, filter(s) in deeplayers are just multiplier of smallest
                     64 for default based on the original paper
    name          -- model name

    Returns:
    model         -- a Model() instance in Keras
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # Contracting Path
    # contracting layer 1 (64 filters)
    conv1 = Conv2D(start_neurons * 1, 3, activation = "relu", padding = "same")(X_input)
    conv1 = Conv2D(start_neurons * 1, 3, activation = "relu", padding = "same")(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # contracting layer 2 (128 filters)
    conv2 = Conv2D(start_neurons * 2, 3, activation = "relu", padding = "same")(pool1)
    conv2 = Conv2D(start_neurons * 2, 3, activation = "relu", padding = "same")(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # contracting layer 3 (256 filters)
    conv3 = Conv2D(start_neurons * 4, 3, activation = "relu", padding = "same")(pool2)
    conv3 = Conv2D(start_neurons * 4, 3, activation = "relu", padding = "same")(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # contracting layer 4 (512 filters)
    conv4 = Conv2D(start_neurons * 8, 3, activation = "relu", padding = "same")(pool3)
    conv4 = Conv2D(start_neurons * 8, 3, activation = "relu", padding = "same")(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle Layer (1024 filters)
    convm = Conv2D(start_neurons * 16, 3, activation = "relu", padding = "same")(pool4)
    convm = Conv2D(start_neurons * 16, 3, activation = "relu", padding = "same")(convm)

    # Expansive Path
    # expansive layer 4 (512 filters)
    deconv4 = Conv2DTranspose(start_neurons * 8, 3, strides = (2, 2), padding = "same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, 3, activation = "relu", padding = "same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, 3, activation = "relu", padding = "same")(uconv4)

    # expansive layer 3 (256 filters)
    deconv3 = Conv2DTranspose(start_neurons * 4, 3, strides = (2, 2), padding = "same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, 3, activation = "relu", padding = "same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, 3, activation = "relu", padding = "same")(uconv3)

    # expansive layer 2 (128 filters)
    deconv2 = Conv2DTranspose(start_neurons * 2, 3, strides = (2, 2), padding = "same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, 3, activation = "relu", padding = "same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, 3, activation = "relu", padding = "same")(uconv2)

    # expansive layer 1 (64 filters)
    deconv1 = Conv2DTranspose(start_neurons * 1, 3, strides = (2, 2), padding = "same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, 3, activation = "relu", padding = "same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, 3, activation = "relu", padding = "same")(uconv1)

    # Output layer
    output_layer = Conv2D(2, 3, activation = "relu", padding = "same")(uconv1)
    output_layer = Conv2D(1, 1, activation = "sigmoid")(output_layer)

    model = Model(inputs = X_input, outputs = output_layer, name = name)
    return model
