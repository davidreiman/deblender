import tensorflow as tf
from .layers import *


class Model:
    """
    Abstract neural network model class.
    """
    def __call__(self, x, reuse=False):
        """
        Forward pass through network.

        Args:
            x(tf.Tensor): input tensor.
            name(str): name for variable scope definition.
        """
        return NotImplementedError("Abstract class methods should not be called.")

    @property
    def vars(self):
        """
        Getter function for model variables.
        """
        return NotImplementedError("Abstract class methods should not be called.")


class ResNet(Model):
    def __init__(self, num_blocks, name='resnet'):
        self.name = name
        self.num_blocks = num_blocks

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

        x = conv_2d(
            x,
            kernel_size=3,
            filters=64,
            stride=1,
            activation='prelu'
        )

        x_ = tf.identity(x)

        for i in range(self.num_blocks):

            x = res_block_2d(
                x,
                kernel_size=3,
                activation='prelu'
            )

        x = tf.add(x, x_)

        for j in range(2):
            x = subpixel_conv(
                x,
                upscale_ratio=2,
                activation='prelu'
            )

        x = conv_2d(
            x,
            kernel_size=3,
            filters=3,
            stride=1,
            activation='sigmoid'
        )

        return x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
