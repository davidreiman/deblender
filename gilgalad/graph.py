import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm as pbar

from .utils import *


class BaseGraph:

    """ Abstract Graph class to train and evaluate models. """

    def build_graph(self, params=None):
        """
        Builds the graph with hyperparameters specified by params dictionary.

        Args:
            n_batches(int): number of batchwise update iterations.
        """
        raise NotImplementedError('Abstract class methods should not be called.')

    def train(self, n_batches, summary_interval, ckpt_interval):
        """
        Trains the model for n_batches update iterations.

        Args:
            n_batches(int): number of batchwise update iterations.
        """
        raise NotImplementedError('Abstract class methods should not be called.')

    def evaluate(self):
        """
        Evaluates the model on a validation dataset.
        """
        raise NotImplementedError('Abstract class methods should not be called.')

    def save(self):
        """
        Saves model to file.
        """
        raise NotImplementedError('Abstract class methods should not be called.')

    def summarize(self):
        """
        Pushes summaries to TensorBoard via log file.
        """
        raise NotImplementedError('Abstract class methods should not be called.')


class Graph(BaseGraph):

    def __init__(self, network, sampler, logdir=None, ckptdir=None):
        """
        Builds graph and defines loss functions & optimizers.

        Args:
            network(models.Model): neural network model.
            sampler(utils.DataSampler): data sampler object.
            logdir(str): filepath for TensorBoard logging.
            ckptdir(str): filepath for saving model.
        """

        self.network = network
        self.data = sampler
        self.logdir = logdir
        self.ckptdir = ckptdir

        self.build_graph()

    def build_graph(self, params=None):
        if hasattr(self, 'sess'):
            self.sess.close()

        tf.reset_default_graph()
        self.data.initialize()

        self.x, self.y = self.data.get_batch()

        self.y_ = self.network(self.x, params=params)

        self.loss = tf.losses.mean_squared_error(self.y, self.y_)
        self.eval_metric = tf.losses.mean_squared_error(self.y, self.y_)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(
                learning_rate=params['lr'] if params else 0.001
            )

            self.update = opt.minimize(
                loss=self.loss,
                var_list=self.network.vars,
                global_step=self.global_step
            )

        if self.logdir and not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if self.ckptdir and not os.path.isdir(self.ckptdir):
            os.makedirs(self.ckptdir)

        loss_summary = tf.summary.scalar("Loss", self.loss)
        image_summary = tf.summary.image("Output", self.y_)
        self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if self.logdir:
            self.summary_writer = tf.summary.FileWriter(
                logdir=self.logdir,
                graph=self.sess.graph
            )

    def save(self):
        if self.ckptdir:
            self.saver.save(
                sess=self.sess,
                save_path=os.path.join(self.ckptdir, 'ckpt'),
                global_step=self.global_step
            )

    def summarize(self):
        if self.logdir:
            summaries = self.sess.run(self.merged_summary)
            global_step = self.sess.run(self.global_step)
            self.summary_writer.add_summary(
                summary=summaries,
                global_step=global_step
            )

    def train(self, n_batches, summary_interval=100, ckpt_interval=10000):
        self.network.training = True
        self.sess.run(self.data.get_dataset('train'))

        try:
            for batch in pbar(range(n_batches), unit='batch'):
                self.sess.run(self.update)

                if batch % summary_interval == 0:
                    self.summarize()

                if batch % ckpt_interval == 0 or batch + 1 == n_batches:
                    self.save()

        except KeyboardInterrupt:
            print("Saving model before quitting...")
            self.save()
            print("Save complete. Training stopped.")

        finally:
            loss = self.sess.run(self.loss)
            return loss

    def evaluate(self):
        self.network.training = False
        self.sess.run(self.data.get_dataset('valid'))

        scores = []
        while True:
            try:
                metric = self.sess.run(self.eval_metric)
                scores.append(metric)
            except tf.errors.OutOfRangeError:
                break

        mean_score = np.mean(scores)
        return mean_score

    def infer(self):
        self.network.training = False
        self.sess.run(self.data.get_dataset('test'))
        pass
