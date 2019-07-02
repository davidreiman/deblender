import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import warnings
import tensorflow as tf

from tqdm import tqdm
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import AffineTransform, rescale, warp, rotate

from deblender.utils import np_to_tfrecords


plt.rcParams['figure.figsize'] = [8, 8]
warnings.filterwarnings('ignore')


def crop(x, h=240, w=240):
    assert x.shape[0] >= h, x.shape[1] >= w
    ch, cw = int((x.shape[0]-h)/2), int((x.shape[1]-w)/2)
    return x[ch:x.shape[0]-ch, cw:x.shape[1]-cw, :]


def downsample(x, factor=3.):
    return rescale(x, 1./factor, mode='constant')


def crop_and_downsample(x):
    return downsample(crop(x))


def perturb(x):
    sx, sy = np.array(x.shape[:2])//2

    rotation = np.random.uniform(0, 2*np.pi)
    scale = np.power(np.e, np.random.uniform(-1, 0.5))
    v_flip = np.random.choice([True, False])
    h_flip = np.random.choice([True, False])
    shift = np.concatenate([np.arange(-50, -10), np.arange(10, 50)])
    translation = np.random.choice(shift, size=2)

    if v_flip:
        x = x[::-1, :, :]
    if h_flip:
        x = x[:, ::-1, :]

    shift = AffineTransform(translation=[-sx, -sy])
    inv_shift = AffineTransform(translation=[sx, sy])

    tform = AffineTransform(
        scale=[scale, scale],
        rotation=rotation,
        translation=translation
    )

    return warp(x, (shift + tform + inv_shift).inverse)


def merge(x1, x2):
    assert np.shape(x1) == np.shape(x2)
    y = [np.max(np.dstack([x1[:, :, i], x2[:, :, i]]), -1)
        for i in range(x1.shape[-1])]
    return np.dstack(y)


def get_batch(batch_size, training=True):
    if training:
        filepath = "/vol/data/unprocessed/galaxy-zoo-images/train"
    else:
        filepath = "/vol/data/unprocessed/galaxy-zoo-images/test"

    print('Generating data from: {}'.format(filepath))

    files = np.random.choice(os.listdir(filepath), size=2*batch_size)
    images = [img_as_float(imread(os.path.join(filepath, file)))
        for file in files]

    x = [crop_and_downsample(image) for image in images[:batch_size]]
    y = [crop_and_downsample(perturb(image)) for image in images[batch_size:]]

    merged = [merge(x1, x2) for x1, x2 in zip(x, y)]

    x = 2*np.array(x) - 1
    y = 2*np.array(y) - 1
    merged = np.array(merged)

    x = x.reshape(batch_size, -1)
    y = y.reshape(batch_size, -1)
    merged = merged.reshape(batch_size, -1)

    return x, y, merged


def plot_batch(batch):
    batch_size = len(batch[0])
    columns = len(batch)

    fig, axes = plt.subplots(
        columns,
        batch_size,
        figsize=(2*batch_size, 2*columns)
    )

    for i in range(columns):
        for j in range(batch_size):
            ax = axes[i][j]
            ax.imshow(batch[i][j])
            ax.set_aspect('equal')
            ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def main():
    tfrecords_path = '/vol/data/deblender/train'

    if not os.path.isdir(tfrecords_path):
        os.makedirs(tfrecords_path)

    for i in tqdm(range(1000)):

        x, y, z = get_batch(320, training=False)

        filename = os.path.join(tfrecords_path, 'test-batch_{}'.format(i))
        np_to_tfrecords(z, x, y, filename)


if __name__ == '__main__':
    main()
