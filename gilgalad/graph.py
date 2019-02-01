import numpy as np
import tensorflow as tf
from tqdm import tqdm as pbar


class BaseGraph:

    """ Abstract Graph class to train and evaluate models. """

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


class Graph(BaseGraph):

    def __init__(self, network, sampler, logdir=None, ckptdir=None):
        """
        Builds graph and defines loss functions & optimizers.

        Args:
            network(models.Model): neural network model.
            sampler(utils.DataSampler): data sampler object.
            logdir(str): filepath location for TensorBoard logging.
            ckptdir(str): filepath location for saving model.
        """

        self.network = network
        self.data = sampler
        self.logdir = logdir
        self.ckptdir = ckptdir

        self.x, self.y, self.z = self.data.get_batch()

        self.y_ = self.network(self.x)

        self.loss = tf.losses.mean_squared_error(self.y, self.y_)
        self.eval_metric = tf.metrics.mean_absolute_error(self.y, self.y_)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer().minimize(
                loss=self.loss,
                var_list=self.network.vars,
                global_step=self.global_step
            )

        if self.logdir and not os.path.isdir(logdir):
            os.makedirs(logdir)
        if self.ckptdir and not os.path.isdir(ckptdir):
            os.makedirs(ckptdir)

        self.global_step = tf.Variable(1, trainable=False)

        loss_summary = tf.summary.scalar("Loss", self.loss)
        self.merged_summary = tf.summary.merge_all()

        if self.logdir:
            self.summary_writer = tf.summary.FileWriter(logdir)
        self.saver = tf.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

    def save(self):
        if self.ckptdir:
            self.saver.save(
                sess=self.sess,
                save_path=os.path.join(self.ckptdir, 'ckpt'),
                global_step=self.global_step
            )

    def add_summary(self):
        if self.logdir:
            summaries = self.sess.run(self.merged_summary)
            self.summary_writer.add_summary(
                summary=summaries,
                global_step=self.global_step
            )

    def train(self, n_batches, summary_interval=100, ckpt_interval=10000):
        self.network.training = True
        self.sess.run(self.data.initialize('train'))

        try:
            for batch in pbar(range(n_batches), unit='batch'):
                self.sess.run(self.opt)

                if batch % summary_interval == 0:
                    self.add_summary()

                if batch % ckpt_interval == 0 or batch + 1 == n_batches:
                    self.save()

        except KeyboardInterrupt:
            print("Saving model before quitting...")
            self.save()
            print("Save complete. Training stopped.")

    def evaluate(self):
        self.network.training = False
        self.sess.run(self.data.initialize('valid'))

        scores = []
        while True:
            try:
                metric = sess.run(self.eval_metric)
                scores.append(metric)
            except tf.errors.OutOfRangeError:
                break

        mean_score = np.mean(scores)

        return mean_score

    def infer(self):
        self.network.training = False
        self.sess.run(self.data.initialize('test'))
        pass
