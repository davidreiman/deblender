import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DataSampler:
    """
    Creates TensorFlow Dataset objects from directories containing
    .tfrecord TensorFlow binaries and passes tensors to graph. The
    resulting sampler is reinitializable onto any of three datasets
    (training, validation, testing) via the initialize method.

    Args:
        train_path(str): training data filepath containing .tfrecords files.
        train_path(str): validation data filepath containing .tfrecords files.
        train_path(str): test data filepath containing .tfrecords files.
        data_shapes(dict): data shape dictionary to specify reshaping operation.
        batch_size(int): number of samples per batch call.
        shuffle(bool): shuffle data (only applicable to training set).
        buffer_size(int): size of shuffled buffer TFDataset will draw from.

    Note: this class currently only supports float data. In the future, it will
    need to accomodate integer-valued data as well.
    """
    def __init__(self, train_path, valid_path, test_path, data_shapes,
        batch_size, shuffle=True, buffer_size=10000):
        assert isinstance(batch_size, int), "Batch size must be integer-valued."
        assert isinstance(buffer_size, int), "Buffer size must be integer-valued."

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.data_shapes = data_shapes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.initialized = False

    def initialize(self):
        valid, test = map(self.make_dataset, [self.valid_path, self.test_path])
        train = self.make_dataset(self.train_path, train=True)

        self.iter = tf.data.Iterator.from_structure(
            train.output_types, train.output_shapes)
        train_init, valid_init, test_init = map(
            self.iter.make_initializer, [train, valid, test])
        self.init_ops = dict(zip(['train', 'valid', 'test'],
            [train_init, valid_init, test_init]))
        self.initialized = True

    def make_dataset(self, filepath, train=False):
        files = [os.path.join(filepath, file) for file
            in os.listdir(filepath) if file.endswith('.tfrecords')]
        dataset = tf.data.TFRecordDataset(files).map(self.decoder)

        if train:
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=self.buffer_size)
            return dataset.repeat().batch(self.batch_size)
        else:
            return dataset.batch(self.batch_size)

    def get_dataset(self, dataset='train'):
        if not self.initialized:
            raise ValueError('Sampler must be initialized before dataset retrieval.')
        try:
            return self.init_ops.get(dataset)
        except:
            raise ValueError('Dataset unknown or unavailable.')

    def decoder(self, example_proto):
        feature_keys = {k: tf.FixedLenFeature(np.prod(v), tf.float32)
            for k, v in self.data_shapes.items()}
        parsed_features = tf.parse_single_example(example_proto, feature_keys)
        parsed = [parsed_features[key] for key in self.data_shapes.keys()]
        return parsed

    def get_batch(self):
        if not self.initialized:
            raise ValueError('Sampler must be initialized before batch retrieval.')
        batch = self.iter.get_next()
        batch = [tf.reshape(batch[i], [-1] + list(v))
            for i, v in enumerate(self.data_shapes.values())]
        return batch


def np_to_tfrecords(data, file_path_prefix, verbose=False):
    """
    Converts 2-D NumPy arrays to TensorFlow binaries.

    Args:
        data(dict): dictionary of NumPy arrays and corresponding keys.
        file_path_prefix(str): file path for storing resulting .tfrecords.
        verbose(bool): function verbosity for debugging.

    Note that the keys provided to create .tfrecord files must correspond with
    the keys passed to the utils.DataSampler class in the data_shapes arg.

    Adapted from a Gist by Sangwoong Yoon.
    """
    def _dtype_feature(ndarray):
        """ Match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(
                float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64 or dtype_ == np.int32:
            return lambda array: tf.train.Feature(
                int64_list=tf.train.Int64List(value=array))
        else:
            raise TypeError("The input should be numpy ndarray. \
                               Instead got {}".format(ndarray.dtype))

    feature_types, records = {}, []
    for k, v in data.items():
        assert isinstance(v, np.ndarray)
        n_records = v.shape[0]
        records.append(n_records)
        if not len(v.shape) == 2:
            data[k] = v = v.reshape([n_records, -1])
        feature_types[k] = _dtype_feature(v)

    assert all(x == records[0] for x in records), \
        "All data must have the same number of samples."

    """ Generate TFRecord writer. """
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(
            n_records, result_tf_file))

    """ Iterate over each sample and serialize it as ProtoBuf. """
    for idx in range(n_records):
        d_feature = {}
        for k, v in data.items():
            d_feature[k] = feature_types[k](v[idx])

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))


def restore_session(sess, ckptdir):
    """
    Restores the checkpoint session from disk.
    """
    meta_graph = [os.path.join(ckptdir, file) for file
        in os.listdir(ckptdir) if file.endswith('.meta')][0]
    restorer = tf.train.import_meta_graph(meta_graph)
    restorer.restore(sess, tf.train.latest_checkpoint(ckptdir))


def get_total_params():
    """
    Computes the total number of learnable variables in default graph.
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def get_trainable_params():
    """
    Analyzes trainable variables in default graph.
    """
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def to_stdout(obj):
    """
    Prints arbitrarily long dictionaries or lists to stdout.
    """
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print(k)
                to_stdout(v)
            else:
                print('%s: %s' % (k, v))
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                to_stdout(v)
            else:
                print(v)
    else:
        print(obj)
