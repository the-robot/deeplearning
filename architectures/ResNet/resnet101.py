# Tensorflow v.2.3.1

from block import block

from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def ResNet101(input_shape: typing.Tuple[int] = (64, 64, 3), classes: int = 6) -> Model:
    """
    Please refer to the original paper for more information.

    Implementation of the ResNet101 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1
        -> CONVBLOCK -> IDBLOCK * 2         // conv2_x
        -> CONVBLOCK -> IDBLOCK * 3         // conv3_x
        -> CONVBLOCK -> IDBLOCK * 22        // conv4_x
        -> CONVBLOCK -> IDBLOCK * 2         // conv5_x
        -> AVGPOOL
        -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes     -- integer, number of classes

    Returns:
    model       -- a Model() instance in Keras
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # zero-padding
    X = ZeroPadding2D((3, 3))(X_input)

    # NOTE: conv1
    X = Conv2D(
        filters = 64,
        kernel_size = (7, 7),
        strides = (2, 2),
        name = "conv1",
        kernel_initializer = "glorot_uniform",
    )(X)
    X = BatchNormalization(axis = 3, name = "bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)
