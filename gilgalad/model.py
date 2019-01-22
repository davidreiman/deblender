import tensorflow as tf
from tensorflow import layers as ly
from tensorflow import keras as k

class Model(object):
    """
    Abstract model class for training and evaluating.
    """
    def train(self, epochs):
        """
        Returns a suggestion for parameter values.
        Args:
            parameters (list[sherpa.Parameter]): the parameters.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.
        Returns:
            dict: parameter values.
        """
        raise NotImplementedError("Algorithm class is not usable itself.")

    def load(self, num_trials):
        """
        Reinstantiates the algorithm when loaded.
        Args:
            num_trials (int): number of trials in study so far.
        """
        pass

class Generator:
    def __init__(self, params, num_blocks=8, name='generator'):
        self.params = params
        self.num_blocks = num_blocks
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:

            y = ly.conv1d(
                inputs=x,
                filters=64,
                kernel_size=self.params['g_l1_kern'],
                strides=1,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
                )
            y = k.layers.PReLU()(y)
            y_ = tf.identity(y)

            for i in range(self.num_blocks):
                y = residual_block(y, kern=self.params['res_block_kern'], filters=64)
            
            y = tf.add(y, y_)
            
            y = ly.conv1d(
                inputs=y,
                filters=64,
                kernel_size=self.params['g_l2_kern'],
                strides=10,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
                )
            
            y = ly.conv1d(
                inputs=y,
                filters=64,
                kernel_size=self.params['g_l3_kern'],
                strides=4,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
                )
            
            y = ly.conv1d(
                inputs=y,
                filters=64,
                kernel_size=self.params['g_l4_kern'],
                strides=4,
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
                )

            y = ly.Flatten()(y)
            
            y = ly.dense(
                inputs=y,
                units=512,
                activation=tf.identity,
                )
            y = k.layers.PReLU()(y)
            y = ly.dense(
                inputs=y,
                units=280,
                activation=None,
                )
            
            y = tf.reshape(y, [-1, 280, 1])

            return y

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]