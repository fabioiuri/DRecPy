import math


def dcg(recommendations, relevancies, k=None, strong_relevancy=True):
    """Discounted Cumulative Gain at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevancies: A dict that should map recommended ids to their relevancy value.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.
        strong_relevancy: An optional boolean indicating which variant of the DCG is used. If set to True,
            usually results in smaller values than when it's set to False. Default: True.

    Returns:
        The computed Discounted Cumulative Gain value.
    """
    if k is not None: recommendations = recommendations[:k]

    curr_dcg = 0
    for i, r in enumerate(recommendations):
        rel = relevancies[r]

        if strong_relevancy:
            curr_dcg += (2 ** rel - 1) / math.log2(2 + i)
        else:
            curr_dcg += rel / math.log2(2 + i)

    return curr_dcg


def ndcg(recommendations, relevancies, k=None, strong_relevancy=True):
    """Normalized Discounted Cumulative Gain at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevancies: A dict that should map ids to their relevancy value. Note that both relevant and irrelevant
            item identifiers should be present on this dict, so that the ideal recommendation list can be computed.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.
        strong_relevancy: An optional boolean indicating which variant of the DCG is used. If set to True,
            usually results in smaller values than when it's set to False. Default: True.

    Returns:
        The computed Normalized Discounted Cumulative Gain value.
    """
    curr_dcg = dcg(recommendations, relevancies, k=k, strong_relevancy=strong_relevancy)
    best_recommendations = sorted(relevancies.keys(), key=lambda x: -relevancies[x])
    best_dcg = dcg(best_recommendations, relevancies, k=k, strong_relevancy=strong_relevancy)

    return curr_dcg / best_dcg


def hit_ratio(recommendations, relevant_recommendations, k=None):
    """Hit Ratio at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendations: A list with the identifiers of the relevant recommendation items.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.

    Returns:
        The computed Hit Ratio value.
    """
    if k is not None: recommendations = recommendations[:k]

    recommendations = set([str(item) for item in recommendations])
    relevant_recommendations = set([str(item) for item in relevant_recommendations])

    return len(recommendations.intersection(relevant_recommendations)) / len(relevant_recommendations)


def reciprocal_rank(recommendations, relevant_recommendation, k=None):
    """Reciprocal Rank at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendation: The identifier of the most relevant item.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.

    Returns:
        The computed Reciprocal Rank value.
    """
    if k is not None: recommendations = recommendations[:k]

    if relevant_recommendation in recommendations:
        return 1 / (recommendations.index(relevant_recommendation) + 1)
    return 0


def recall(recommendations, relevant_recommendations, k=None):
    """Recall at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendations: A list with the identifiers of the relevant recommendation items.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.

    Returns:
        The computed Recall value.
    """
    if k is not None: recommendations = recommendations[:k]

    in_common = set(recommendations).intersection(set(relevant_recommendations))
    return len(in_common) / len(relevant_recommendations)


def precision(recommendations, relevant_recommendations, k=None):
    """Precision at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendations: A list with the identifiers of the relevant recommendation items.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.

    Returns:
        The computed Precision value.
    """
    if k is not None: recommendations = recommendations[:k]

    in_common = set(recommendations).intersection(set(relevant_recommendations))
    return len(in_common) / len(recommendations)


def f_score(recommendations, relevant_recommendations, k=None, beta=1):
    """F-score at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendations: A list with the identifiers of the relevant recommendation items.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.
        beta: An optional integer representing the weight of the recall value on the combined score.
            Beta > 1 favours recall over precision, while beta < 1 favours precision over recall.
            Default: 1.

    Returns:
        The computed F-score value.
    """
    p = precision(recommendations, relevant_recommendations, k=k)
    r = recall(recommendations, relevant_recommendations, k=k)

    return (1 + beta ** 2) * p * r / ((beta ** 2 * p) + r)


def average_precision(recommendations, relevant_recommendations, k=None):
    """Average Precision at k.

    Args:
        recommendations: A list with the identifiers of the recommended items.
        relevant_recommendations: A list with the identifiers of the relevant recommendation items.
        k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
            truncated after the k-th element. Default: None.

    Returns:
        The computed Average Precision value.
    """
    if k is not None: recommendations = recommendations[:k]

    _sum = 0
    for r, i in zip(recommendations, range(1, len(recommendations) + 1)):
        if r in relevant_recommendations and r not in recommendations[:i-1]:  # r is relevant and the first occurrence
            _sum += precision(recommendations, relevant_recommendations, k=i)

    if k is None: return _sum / len(relevant_recommendations)
    return _sum / min(len(relevant_recommendations), k)
