import os
import tfplot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataSampler:
    def __init__(self, train_filepath, batch_size, valid_filepath=None, test_filepath=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        train_files = [os.path.join(train_filepath, file) for file in os.listdir(train_filepath) if file.endswith('.tfrecords')]
        train_dataset = self.make_dataset(train_files)
        self.iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.train_init_op = self.iter.make_initializer(train_dataset)

        if valid_filepath is not None:
            valid_files = [os.path.join(valid_filepath, file) for file in os.listdir(valid_filepath) if file.endswith('.tfrecords')]
            valid_dataset = self.make_test_dataset(valid_files)
            self.valid_init_op = self.iter.make_initializer(valid_dataset)

        if test_filepath is not None:
            test_files = [os.path.join(test_filepath, file) for file in os.listdir(test_filepath) if file.endswith('.tfrecords')]
            test_dataset = self.make_test_dataset(test_files)
            self.test_init_op = self.iter.make_initializer(test_dataset)
    
    def make_dataset(self, files):
        dataset = tf.data.TFRecordDataset(files).map(self.decoder)
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        return dataset.repeat().batch(self.batch_size)
    
    def make_test_dataset(self, files):
        dataset = tf.data.TFRecordDataset(files).map(self.decoder)
        return dataset.batch(self.batch_size)
    
    def initialize(self, dataset='train'):
        if dataset == 'train':
            return self.train_init_op
        elif dataset == 'valid':
            return self.valid_init_op
        elif dataset == 'test':
            return self.test_init_op
        else:
            raise ValueError('Dataset unknown or unavailable.')

    def decoder(self, example_proto):
        keys_to_features = {'latent' : tf.FixedLenFeature(4000, tf.float32),
                            'target' : tf.FixedLenFeature(280, tf.float32),
                            'metadata' : tf.FixedLenFeature(2, tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['latent'], parsed_features['target'], parsed_features['metadata']

    def get_batch(self):
        x, y, z = self.iter.get_next()
        x = tf.reshape(x, [-1, 4000, 1])
        y = tf.reshape(y, [-1, 280, 1])
        z = tf.reshape(z, [-1, 2])
        return x, y, z

    
def plot_spectrum(spec):
    plt.style.use('seaborn')
    fig, ax = tfplot.subplots(figsize=(4, 3))
    im = ax.plot(np.arange(1210, 1280, 0.25), spec)
    return fig


def get_total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters