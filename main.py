import gilgalad as gg

data_shapes = {
    'x': (32, 32, 3),
    'y': (128, 128, 3),
}

sampler = gg.utils.DataSampler(
    train_path='data/train',
    valid_path='data/valid',
    test_path='data/test',
    data_shapes=data_shapes,
    batch_size=1
)

network = gg.models.ResNet()

logdir = '/vol/projects/deblender/logdir'
ckptdir = '/vol/projects/deblender/ckptdir'

hyperparameters = {
    'Discrete':
        {'filters': [64, 128],
         'kernel_size': [3, 5]},
    'Continuous':
        {'lr': [1e-5, 1e-3]},
    'Choice':
        {'activation': ['relu', 'prelu']}
}

graph = gg.graph.Graph(network, sampler, logdir, ckptdir)
graph.train()
