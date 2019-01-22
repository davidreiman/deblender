import numpy as np
import gilgalad as gg
import tensorflow as tf


class ResNet:
    def __init__(self, num_blocks, name='resnet'):
        self.name = name
        self.num_blocks = num_blocks

    def __call__(self, x):
        x = gg.layers.conv_2d(x, kernel_size=3, filters=64, stride=1, activation='prelu')

        x_ = tf.identity(x)

        for i in range(self.num_blocks):
            x = gg.layers.res_block_2d(x, kernel_size=3, activation='prelu')

        x = tf.add(x, x_)

        for j in range(2):
            x = gg.layers.subpixel_convolution(x, upscale_ratio=2, activation='prelu')

        x = gg.layers.conv_2d(x, kernel_size=3, filters=3, stride=1, activation='sigmoid')

        return x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


def build_graph():

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = net(x)

    print(y.shape.as_list())


net = ResNet(5)
build_graph()
