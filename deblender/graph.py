import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm as pbar

from .utils import *
from .plotting import make_plot


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
    def __init__(self, generator, discriminator, vgg, sampler,
        logdir=None, ckptdir=None):
        """
        Builds graph and defines loss functions & optimizers.

        Args:
            network(models.Model): neural network model.
            sampler(utils.DataSampler): data sampler object.
            logdir(str): filepath for TensorBoard logging.
            ckptdir(str): filepath for saving model.
        """

        self.g = generator
        self.d = discriminator
        self.data = sampler
        self.vgg = vgg
        self.logdir = logdir
        self.ckptdir = ckptdir

        self.build_graph()

    def build_graph(self, params=None):
        if hasattr(self, 'sess'):
            self.sess.close()

        tf.reset_default_graph()

        self.vgg.initialize()
        self.data.initialize()

        self.blended, self.x1, self.x2 = self.data.get_batch()

        self.y1, self.y2 = self.g(self.blended)

        logits_real_1 = self.d(self.x1, reuse=False)
        logits_real_2 = self.d(self.x2)
        logits_fake_1 = self.d(self.y1)
        logits_fake_2 = self.d(self.y2)

        d_loss_real = (tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_real_1), logits_real_1)
            + tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_real_2), logits_real_2))

        d_loss_fake = (tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(logits_fake_1), logits_fake_1)
            + tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(logits_fake_2), logits_fake_2))

        self.d_loss = 0.5*(d_loss_real + d_loss_fake)

        g_loss = (tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_fake_1), logits_fake_1)
            + tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_fake_2), logits_fake_2))

        mse_loss = (tf.losses.mean_squared_error(self.x1, self.y1)
            + tf.losses.mean_squared_error(self.x2, self.y2))

        self.vgg_loss = (tf.losses.mean_squared_error(
            self.vgg(self.x1), self.vgg(self.y1))
            + tf.losses.mean_squared_error(
            self.vgg(self.x2), self.vgg(self.y2)))

        self.g_loss = 0.5*(mse_loss + 1e-3*g_loss)

        self.lr = tf.Variable(1e-4, trainable=False)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(self.d_loss, self.global_step, self.d.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(self.g_loss, self.global_step, self.g.vars)
            self.vgg_adam = tf.train.AdamOptimizer(learning_rate=self.lr) \
                .minimize(self.vgg_loss, self.global_step, self.g.vars)

        if self.logdir and not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if self.ckptdir and not os.path.isdir(self.ckptdir):
            os.makedirs(self.ckptdir)

        image_summary_blended = tf.summary.image("Blended_Image", self.blended)
        image_summary_x1 = tf.summary.image("Image_X1", self.x1)
        image_summary_x2 = tf.summary.image("Image_X2", self.x2)
        image_summary_y1 = tf.summary.image("Image_Y1", self.y1)
        image_summary_y2 = tf.summary.image("Image_Y2", self.y2)

        loss_summary_g = tf.summary.scalar("Generator_Loss", self.g_loss)
        loss_summary_d = tf.summary.scalar("Discriminator_Loss", self.d_loss)
        loss_summary_vgg = tf.summary.scalar("VGG_Loss", self.vgg_loss)
        lr_summary = tf.summary.scalar("Learning_Rate", self.lr)

        self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

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

    def lr_drop(self):
        current_lr = self.sess.run(self.lr)
        self.sess.run(self.lr.assign(current_lr/10))

    def train(self, n_batches=200000, n_burn_in_batches=2000,
        summary_interval=100, ckpt_interval=10000, progress_bar=True):

        self.sess.run(self.data.get_dataset('train'))
        self.g.training = True; self.d.training = True

        burn_in_iter = range(n_burn_in_batches)
        standard_iter = range(n_batches)

        if progress_bar:
            burn_in_iter = pbar(burn_in_iter, unit='batch')
            standard_iter = pbar(standard_iter, unit='batch')

        try:
            for batch in burn_in_iter:
                self.sess.run(self.vgg_adam)

            for batch in standard_iter:
                self.sess.run(self.d_adam)
                self.sess.run(self.g_adam)

                if batch % 100 == summary_interval:
                    self.summarize()

                if batch == batches//2:
                    self.lr_drop()

                if batch % ckpt_interval == 0 or batch + 1 == batches:
                    self.save()

        except KeyboardInterrupt:
            print("Saving model before quitting...")
            self.save()
            print("Save complete. Training stopped.")

        finally:
            losses = self.sess.run([self.g_loss, self.d_loss])
            return losses

    def image_evaluate(self, savedir='/data/predictions'):
        self.sess.run(self.data.get_dataset('valid'))
        self.g.training = False; self.d.training = False

        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        batch = 0
        while True:
            try:
                blended, true_x, true_y, gan_x, gan_y = self.sess.run(
                    [self.blended, self.x1, self.x2, self.y1, self.y2]
                )

                true_x, true_y, gan_x, gan_y = map(
                    lambda x: (x + 1)/2, [true_x, true_y, gan_x, gan_y]
                )

                make_plots(blended, true_x, true_y, gan_x, gan_y, savedir, batch)
                batch += 1

            except tf.errors.OutOfRangeError:
                break

    def metric_evaluate(self):
        self.sess.run(self.data.get_dataset('valid'))
        self.g.training = False; self.d.training = False

        psnr, ssim = [], []
        while True:
            try:
                blended, true_x, true_y, gan_x, gan_y = self.sess.run(
                    [self.blended, self.x1, self.x2, self.y1, self.y2]
                )

                true_x, true_y, gan_x, gan_y = map(
                    lambda x: (x + 1)/2, [true_x, true_y, gan_x, gan_y]
                )

                for i in range(self.data.batch_size):
                    psnrs.append(
                        compare_psnr(im_test=gan_x[i], im_true=true_x[i]))
                    psnrs.append(
                        compare_psnr(im_test=gan_y[i], im_true=true_y[i]))
                    ssims.append(
                        compare_ssim(X=gan_x[i], Y=true_x[i], multichannel=True))
                    ssims.append(
                        compare_ssim(X=gan_y[i], Y=true_y[i], multichannel=True))

            except tf.errors.OutOfRangeError:
                break

            finally:
                return psnr, ssim
