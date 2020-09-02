from DRecPy.Evaluation.Metrics import PredictiveMetricABC
from DRecPy.Evaluation.Metrics import RMSE
from DRecPy.Evaluation.Metrics import MSE
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
        metrics: An optional list containing instances of PredictiveMetricABC. Default: [RMSE(), MSE()].
        verbose: A boolean indicating whether state logs should be produced or not. Default: true.

    Returns:
        A dict containing each metric name mapping to the corresponding metric value.
    """
    if ds_test is None: ds_test = model.interaction_dataset
    if n_test_predictions is None: n_test_predictions = len(ds_test)

    assert n_test_predictions > 0, f'The number of test users ({n_test_predictions}) should be > 0.'

    metrics = kwds.get('metrics', [RMSE(), MSE()])

    assert isinstance(metrics, list), f'Expected "metrics" argument to be a list and found {type(metrics)}. ' \
        f'Should contain instances of PredictiveMetricABC.'

    for m in metrics:
        assert isinstance(m, PredictiveMetricABC), f'Expected metric {m} to be an instance of type PredictiveMetricABC.'

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
    metric_values = {m.name: round(m(y_true, y_pred), 4) for m in metrics}

    return metric_values
