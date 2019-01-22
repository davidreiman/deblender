from tensorflow import keras as k
from tensorflow import layers as ly


def residual_block(x, kern, filters):
    
    y = ly.conv1d(
        inputs=x,
        filters=filters,
        kernel_size=kern,
        strides=1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        activation=tf.identity
        )
    y = k.layers.PReLU()(y)
    y = ly.conv1d(
        inputs=y,
        filters=filters,
        kernel_size=kern,
        strides=1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        activation=tf.identity
        )
    
    return tf.add(x, y)


def subpixel_convolution(x, block_size):
    
    y = ly.conv1d(
    inputs=x,
    filters=256,
    kernel_size=3,
    strides=1,
    padding='same',
    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
    activation=tf.identity
    )
    y = tf.depth_to_space(y, block_size=block_size)
    
    return k.layers.PReLU()(y)


def convolutional_block(x, filters, stride):
    
    y = ly.conv1d(
    inputs=x,
    filters=filters,
    kernel_size=3,
    strides=stride,
    padding='same',
    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
    activation=tf.identity
    )
    y = ly.batch_normalization(y)
    
    return k.layers.LeakyReLU(alpha=0.2)(y)