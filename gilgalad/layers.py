import tensorflow as tf
from tensorflow import keras as k
from tensorflow import layers as ly


nonlinear = {
    'tanh': tf.tanh,
    'sigmoid': tf.sigmoid,
    'linear': tf.identity,
    'relu': tf.nn.relu,
    'leaky_relu': k.layers.LeakyReLU(alpha=0.2),
    'prelu': k.layers.PReLU(),
    'softmax': tf.nn.softmax,
}


def res_block_1d(x, kernel_size, activation, batch_norm=True):

    activation = activation.lower()
    assert len(x.shape) == 3, "Input tensor must be 3-dimensional."

    filters = int(x.shape[2])

    y = ly.conv1d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
        )

    if batch_norm:
        y = ly.batch_normalization(y)

    y = nonlinear[activation](y)

    y = ly.conv1d(
        inputs=y,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
        )

    if batch_norm:
        y = ly.batch_normalization(y)

    return tf.add(x, y)


def res_block_2d(x, kernel_size, activation, batch_norm=True):

    activation = activation.lower()
    assert len(x.shape) == 4, "Input tensor must be 4-dimensional."

    filters = int(x.shape[3])

    y = ly.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
        )

    if batch_norm:
        y = ly.batch_normalization(y)

    y = nonlinear[activation](y)

    y = ly.conv2d(
        inputs=y,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
        )

    if batch_norm:
        y = ly.batch_normalization(y)

    return tf.add(x, y)


def subpixel_convolution(x, upscale_ratio, activation, kernel_size=3, stride=1):

    activation = activation.lower()
    assert len(x.shape) == 4, "Input tensor must be 4-dimensional."

    n_filters = int(x.shape[3])

    y = ly.conv2d(
    inputs=x,
    filters=n_filters*upscale_ratio**2,
    kernel_size=kernel_size,
    strides=stride,
    padding='same'
    )

    y = tf.depth_to_space(y, block_size=upscale_ratio)

    y = nonlinear[activation](y)

    return y


def conv_block_1d(x, kernel_size, filters, stride, activation,
        batch_norm=True):

    activation = activation.lower()

    y = ly.conv1d(
    inputs=x,
    filters=filters,
    kernel_size=kernel_size,
    strides=stride,
    padding='same'
    )

    y = nonlinear[activation](y)

    if batch_norm:
        y = ly.batch_normalization(y)

    return y


def conv_block_2d(x, kernel_size, filters, stride, activation,
        batch_norm=True):

    activation = activation.lower()

    y = ly.conv2d(
    inputs=x,
    filters=filters,
    kernel_size=kernel_size,
    strides=stride,
    padding='same'
    )

    y = nonlinear[activation](y)

    if batch_norm:
        y = ly.batch_normalization(y)

    return y


def conv_2d(x, kernel_size, filters, stride, activation):

    activation = activation.lower()

    y = ly.conv2d(
    inputs=x,
    filters=filters,
    kernel_size=kernel_size,
    strides=stride,
    padding='same'
    )

    y = nonlinear[activation](y)

    return y
