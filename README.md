<p align="center">
  <img src="docs/images/logo.png">
</p>

---

<p align="center">
My personal deep learning library with frequently used utilities and layer combinations.
</p>

## Getting Started

Clone the repository, navigate to the local directory and begin building your models in the models.py module.


## Model Selection

Gil-Galad model selection is employed via Sherpa's Bayesian optimization suite with sklearn's Gaussian Process backend. Bayesian optimization specifies a distribution over functions via a kernel function and prior. Here, the mean function corresponds to a surrogate objective function whose predictor variables are the model hyperparameters. The prior distribution over functions is updated via Bayes' rule to account for trial runs wherein the independent variables specify the model and the dependent variable is the evaluation of said model on the validation dataset.

With Gil-Galad, we specify which hyperparameters we will optimize by passing a parameter dictionary to our model class while also defining default hyperparameters during model construction as follows:

```python

y = conv_2d(
  x=x,
  filters=params['filt1'] if params else 64,
  kernel_size=params['kern1'] if params else 3,
  strides=2,
  activation='relu'
)

```
