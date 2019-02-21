import os
import time
import sherpa
from tqdm import tqdm as pbar
from progress.bar import Bar
from .utils import to_stdout


def bayesian_optimization(graph, params, max_trials, iter_per_trial=10,
    batches_per_iter=1000, dashboard=False):
    """
    Model selection via Bayesian optimization.

    Args:
        graph(graph.Graph): an instance of the graph class.
        params(dict): dictionary specifying hyperparameters to optimize.
        max_trials(int): maximum number of Bayesian optimization trials.
        iter_per_trial(int): number of observations to add per trial.
        batches_per_iter(int): number of batches to train per observation.
        dashboard(bool): launch optimization dashboard.

    params = {
        'Discrete':
            {'num_units': [50, 200],
             'kernel_size': [3, 11]},
        'Continuous':
            {'lr': [1e-5, 1e-3]},
        'Choice':
            {'activation': ['relu', 'prelu']},
        'Ordinal':
            {'batch_size': [16, 32, 64]},
    }
    """
    assert isinstance(max_trials, int), "Max trials must be an integer."
    assert isinstance(iter_per_trial, int), "Iters/trial must be an integer."
    assert isinstance(batches_per_iter, int), "Batches/iter must be an integer."

    param_class = {
        'discrete': sherpa.Discrete,
        'continuous': sherpa.Continuous,
        'choice': sherpa.Choice,
        'ordinal': sherpa.Ordinal
    }

    parameters = []
    for k, v in params.items():
        if k.lower() not in param_class.keys():
            raise ValueError("Parameter type not recognized: {}".format(k))
        for key, value in v.items():
            assert isinstance(value, list), "Parameter range must be a list."
            parameters.append(param_class.get(k.lower())(key, value))

    alg = sherpa.algorithms.BayesianOptimization(max_num_trials=max_trials+1)

    study = sherpa.Study(
        parameters=parameters,
        algorithm=alg,
        lower_is_better=True,
        disable_dashboard=not dashboard,
    )

    print("\nHyperparameter Optimization\n===========================\n")

    for i, trial in enumerate(study):

        bar = Bar(
            message='Trial {} of {}'.format(i+1, max_trials),
            max=iter_per_trial,
            fill = '◉'
        )

        graph.build_graph(params=trial.parameters)

        for iteration in range(iter_per_trial):

            training_error = graph.train(
                n_batches=batches_per_iter,
                progress_bar=False
            )
            validation_error = graph.evaluate()

            study.add_observation(
                trial=trial,
                iteration=iteration,
                objective=validation_error,
                context={'Training Error': training_error}
            )
            bar.next()

        study.finalize(trial)
        bar.finish()

    optimum = study.get_best_result()

    print("\nOptimal model\n-------------\n")
    to_stdout(optimum)

    return optimum
