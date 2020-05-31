from DRecPy.Evaluation.Metrics import rmse
from DRecPy.Evaluation.Metrics import mse
from tqdm import tqdm


def predictive_evaluation(model, ds_test=None, count_none_predictions=False, n_test_predictions=None, skip_errors=True, **kwds):
    """Executes a predictive evaluation process, where the given model will be evaluated under the provided settings.

    Args:
        model: An instance of a Recommender to be evaluated.
        ds_test: An optional test InteractionDataset. If none is provided, then the test data will be the model
            training data. Evaluating on train data is not ideal for assessing the model's performance.
        count_none_predictions: An optional boolean indicating whether to count none predictions (i.e. when the model
            predicts None, count it as being a 0) or not (i.e. skip that user-item pair). Default: False.
        n_test_predictions: An optional integer representing the number of predictions to evaluate.
            Default: predict for every (user, item) pair on the test dataset.
        skip_errors: A boolean indicating whether to ignore errors produced during the predict calls, or not.
            Default: False.
        metrics: An optional dict mapping the names of the metrics to a tuple containing the metric eval function as the
            first element, and the default arguments to call it as the second element.
            Eg: {'RMSE': (rmse, {beta: 1.2})}. Default: dict with the following metrics: root-mean-squared error and
            mean-squared error.
        verbose: A boolean indicating whether state logs should be produced or not. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    if ds_test is None: ds_test = model.interaction_dataset
    if n_test_predictions is None: n_test_predictions = len(ds_test)

    assert n_test_predictions > 0, f'The number of test users ({n_test_predictions}) should be > 0.'

    metrics = kwds.get('metrics', {
        'RMSE': (rmse, {}),
        'MSE': (mse, {})
    })

    assert type(metrics) is dict, f'Expected "metrics" argument to be of type dict and found {type(metrics)}. ' \
        f'Should map metric names to a tuple containing the corresponding metric function and an extra argument dict.'

    for m in metrics:
        err_msg = f'Expected metric {m} to map to a tuple containing the corresponding metric function and an extra argument dict.'
        assert type(metrics[m]) is tuple, err_msg
        assert callable(metrics[m][0]), err_msg
        assert type(metrics[m][1]) is dict, err_msg

    n_test_predictions = min(n_test_predictions, len(ds_test))
    if kwds.get('verbose', True):
        _iter = tqdm(ds_test.values(['user', 'item', 'interaction'], to_list=True),
                     total=n_test_predictions, desc='Evaluating model predictive performance', position=0, leave=True)
    else:
        _iter = ds_test.values(['user', 'item', 'interaction'], to_list=True)

    num_predictions_made = 0
    y_pred, y_true = [], []
    for user, item, interaction in _iter:
        if num_predictions_made >= n_test_predictions: break  # reach max number of predictions

        pred = model.predict(user, item, skip_errors=skip_errors)
        if pred is None:
            if count_none_predictions:
                num_predictions_made += 1
                y_pred.append(0)
                y_true.append(interaction)
            continue
        y_pred.append(pred)
        y_true.append(interaction)
        num_predictions_made += 1

    # evaluate performance
    metric_values = {}
    for m in metrics:
        metric_fn = metrics[m][0]
        params = {**metrics[m][1], 'y_true': y_true, 'y_pred': y_pred}
        metric_values[m] = round(metric_fn(**params), 4)

    return metric_values
