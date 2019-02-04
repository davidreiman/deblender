<p align="center">
  <img src="docs/images/logo.png">
</p>

---

<p align="center">
A personal deep learning project template with frequently used utilities and layer combinations.
</p>

## Getting Started

Clone the repository, navigate to the local directory and begin building your models in the models.py module. Data should be divided into training, validation and testing sets and placed in .tfrecords file format in separate directories. Data shapes are specified by a dictionary which is subsequently passed to the data sampler during model creation. Note that the data shape dictionary keys must correspond to the same keys used in converting NumPy arrays to .tfrecords files during preprocessing. The data shape values should be tuples sans batch size.


## Model Selection

Gil-Galad model selection is employed via Sherpa's Bayesian optimization suite which utilizes sklearn's Gaussian process module. Bayesian optimization specifies a distribution over functions via a kernel function and prior. Here, the mean function corresponds to a surrogate objective function whose predictor variables are the model hyperparameters. The prior distribution over functions is updated via Bayes' rule to account for trial runs wherein the independent variables specify the model and the dependent variable is the evaluation of said model on the validation dataset.

With Gil-Galad, we specify which hyperparameters we will optimize by passing a parameter dictionary to our graph class while also defining default hyperparameters during graph and model construction as follows:

### Graph-level

```python
class Graph(BaseGraph):

    def __init__(self, network, sampler, logdir=None, ckptdir=None):
    
        self.network = network
        self.data = sampler
        self.logdir = logdir
        self.ckptdir = ckptdir

        self.build_graph()
        
    def build_graph(self, params=None):
    
        tf.reset_default_graph()

        self.x, self.y, self.z = self.data.get_batch()
        
        self.y_ = self.network(self.x, params=params)
        
        self.loss = tf.losses.mean_squared_error(self.y, self.y_)

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
        
```

### Model-level

```python
class Model:
  def __init__(self, name):
    self.name = name
  
  def __call__(self, x, params):
  
    y = conv_2d(
      x=x,
      filters=params['filters'] if params else 64,
      kernel_size=params['kernel_size'] if params else 3,
      strides=2,
      activation=params['activation'] if params else 'relu'
    )
  
    return y

```

We then define the hyperparameter domain type and ranges in a dictionary. This information accompanies the graph object as arguments for the Bayesian optimization function.

```python
hyperparameters = {
    'Discrete':
        {'filters': [64, 128],
         'kernel_size': [3, 5]},
    'Continuous':
        {'lr': [1e-5, 1e-3]},
    'Choice':
        {'activation': ['relu', 'prelu']}
}

best_model = gg.opt.bayesian_optimization(
    graph=graph,
    params=hyperparameters,
    max_trials=50
)
```

