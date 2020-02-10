import math


def dcg(recommendations, relevant_recommendations, relevancies, k=None, strong_relevancy=True):
    """Discounted Cumulative Gain at k
    Example calls:
    >>> dcg([1, 3, 2, 6, 5, 4], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 14.595390756454924
    >>> dcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 13.848263629272981
    >>> dcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=False) == 6.861126688593502
    >>> dcg([4, 10, 20, 30], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 0.0
    """
    if k is not None: recommendations = recommendations[:k]

    curr_dcg = 0
    for i, r in enumerate(recommendations):
        rel = 0

        if r in relevant_recommendations:
            rel_idx = relevant_recommendations.index(r)
            rel = relevancies[rel_idx]

        if strong_relevancy:
            curr_dcg += (2 ** rel - 1) / math.log2(2 + i)
        else:
            curr_dcg += rel / math.log2(2 + i)

    return curr_dcg


def ndcg(recommendations, relevant_recommendations, relevancies, k=None, strong_relevancy=True):
    """Normalized Discounted Cumulative Gain at k
    Example calls:
    >>> ndcg([1, 3, 2, 6, 5, 4], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 1.0
    >>> ndcg([1, 3, 2, 6, 5], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 1.0
    >>> ndcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 0.9488107485678985
    >>> ndcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=False) == 0.9608081943360617
    >>> ndcg([4, 5, 6, 2, 3, 1], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 0.5908974642816868
    >>> ndcg([4, 5], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 0.04322801383665758
    >>> ndcg([4, 10, 20, 30], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], strong_relevancy=True) == 0.0
    >>> ndcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], k=1, strong_relevancy=True) == 1.0
    >>> ndcg([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 6], [3, 2, 3, 1, 2], k=2, strong_relevancy=True) == 0.7789412530088334
    """
    if k is not None: recommendations = recommendations[:k]

    curr_dcg = dcg(recommendations, relevant_recommendations, relevancies, k=k, strong_relevancy=strong_relevancy)
    best_recommendations = sorted(zip(relevant_recommendations, relevancies), key=lambda x: -x[1])
    best_recommendations = [r for r, _ in best_recommendations]
    best_dcg = dcg(best_recommendations, relevant_recommendations, relevancies, k=k, strong_relevancy=strong_relevancy)

    return curr_dcg / best_dcg


def mean_ndcg(recommendations_list, relevant_recommendations_list, relevancies_list, k=None, strong_relevancy=True):
    """Mean Normalized Discounted Cumulative Gain at k
    Example calls:
    >>> mean_ndcg([[1, 3, 2, 6, 5, 4], [4, 10, 20, 30]], [[1, 2, 3, 5, 6], [1, 2, 3, 5, 6]], [[3, 2, 3, 1, 2], [3, 2, 3, 1, 2]]) == 0.5
    """
    _sum = 0
    for recommendations, relevant_recommendations, relevancies in \
            zip(recommendations_list, relevant_recommendations_list, relevancies_list):
        _sum += ndcg(recommendations, relevant_recommendations, relevancies, k=k, strong_relevancy=strong_relevancy)
    return _sum / len(recommendations_list)


def hit_ratio(recommendations, relevant_recommendations, k=None):
    """Hit Ratio at k
    Example calls:
    >>> hit_ratio([1,2,3], [3]) == 1.0
    >>> hit_ratio([1,2,3], [3,2]) == 1.0
    >>> hit_ratio([1,2,3], [3,2,5]) == 0.6666666666666666
    >>> hit_ratio([1,2,3], [3,2], k=2) == 0.5
    >>> hit_ratio([1,2], [3,2,5]) == 0.3333333333333333
    >>> hit_ratio([1,2], [3,2,5], k=2) == 0.3333333333333333
    """
    if k is not None: recommendations = recommendations[:k]

    recommendations = set([str(item) for item in recommendations])
    relevant_recommendations = set([str(item) for item in relevant_recommendations])

    return len(recommendations.intersection(relevant_recommendations)) / len(relevant_recommendations)


def mean_hit_ratio(recommendations_list, relevant_recommendations_list, k=None):
    """Mean Hit Ratio at k
    Example calls:
    >>> mean_hit_ratio([[1,2,3], [1,2,3]], [[3,2], [3,2,5]]) == 0.8333333333333333
    >>> mean_hit_ratio([[3,2,1], [1,2]], [[3,2], [3,2,5]], k=2) == 0.6666666666666666
    """
    _sum = 0
    for recommendations, relevant_recommendations in zip(recommendations_list, relevant_recommendations_list):
        _sum += hit_ratio(recommendations, relevant_recommendations, k=k)
    return _sum / len(recommendations_list)


def reciprocal_rank(recommendations, relevant_recommendation, k=None):
    """Reciprocal Rank at k
    Example calls:
    >>> reciprocal_rank([1,2,3], 1) == 1.0
    >>> reciprocal_rank([1,2,3], 2) == 0.5
    >>> reciprocal_rank([1,2,3], 3) == 0.3333333333333333
    >>> reciprocal_rank([1,2,3], 4) == 0
    >>> reciprocal_rank([1,2,3], 3, k=2) == 0
    >>> reciprocal_rank([1,2,3], 2, k=2) == 0.5
    :param recommendations:
    :param relevant_recommendation:
    :param k:
    :return:
    """
    if k is not None: recommendations = recommendations[:k]

    if relevant_recommendation in recommendations:
        return 1 / (recommendations.index(relevant_recommendation) + 1)
    return 0


def mean_reciprocal_rank(recommendations_list, relevant_recommendation_list, k=None):
    """Mean Reciprocal Rank at k
    Example calls:
    >>> mean_reciprocal_rank([[1,2,3], [1,2,3], [1,2,3], [1,2,3], [3,2,1]], [1, 2, 3, 4, 3]) == 0.5666666666666667
    >>> mean_reciprocal_rank([[1,2,3], [1,2,3], [1,2,3], [1,2,3], [3,2,1]], [1, 2, 3, 4, 3], k=2) == 0.5
    :param recommendations_list:
    :param relevant_recommendation_list:
    :param k:
    :return:
    """
    _sum = 0
    for recommendations, relevant_recommendation in zip(recommendations_list, relevant_recommendation_list):
        _sum += reciprocal_rank(recommendations, relevant_recommendation, k=k)
    return _sum / len(recommendations_list)


def recall(recommendations, relevant_recommendations, k=None):
    """Recall at k
    >>> recall([1,2,3], [1,2,3]) == 1.0
    >>> recall([1,2,3], [2,3]) == 1.0
    >>> recall([1,2,3], [2,3,4]) == 0.6666666666666666
    >>> recall([1,2,3], [4]) == 0.0
    >>> recall([1,2,3], [2,3], k=2) == 0.5
    """
    if k is not None: recommendations = recommendations[:k]

    in_common = set(recommendations).intersection(set(relevant_recommendations))
    return len(in_common) / len(relevant_recommendations)


def precision(recommendations, relevant_recommendations, k=None):
    """Precision at k
    >>> precision([1,2,3], [1,2,3]) == 1.0
    >>> precision([1,2,3], [2,3]) == 0.6666666666666666
    >>> precision([1,2,3], [2,3,4]) == 0.6666666666666666
    >>> precision([1,2,3], [4]) == 0.0
    >>> precision([1,2,3], [2,3], k=2) == 0.5
    """
    if k is not None: recommendations = recommendations[:k]

    in_common = set(recommendations).intersection(set(relevant_recommendations))
    return len(in_common) / len(recommendations)


def f_score(recommendations, relevant_recommendations, k=None, beta=1):
    """F-score at k
    >>> f_score([1], [1, 2, 4, 8]) == 0.4
    >>> f_score([1, 2], [1, 2, 4, 8]) == 0.6666666666666666
    >>> f_score([1, 2, 3], [1, 2, 4, 8]) == 0.5714285714285715
    >>> f_score([1, 2, 3, 4], [1, 2, 4, 8]) == 0.75
    >>> f_score([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 8]) == 0.6666666666666666
    >>> f_score([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 8], k=4) == 0.75
    >>> f_score([1, 2, 4, 8], [1, 2, 4, 8]) == 1.0
    """

    p = precision(recommendations, relevant_recommendations, k=k)
    r = recall(recommendations, relevant_recommendations, k=k)

    return (1 + beta ** 2) * p * r / ((beta ** 2 * p) + r)


def average_precision(recommendations, relevant_recommendations, k=None):
    """Average Precision at k
    >>> average_precision([0,1,1], [1]) == 0.5
    >>> average_precision([0,1,0], [1]) == 0.5
    >>> average_precision([0,0,1], [1]) == 0.3333333333333333
    >>> average_precision([0,0,0], [1]) == 0.0
    >>> average_precision([1,0,0], [1]) == 1.0
    >>> average_precision([1,2,1], [1,2]) == 1.0
    >>> average_precision([2,1,1], [1,2]) == 0.5

    >>> average_precision([1,3,5], [1,3,4,6]) == 0.5
    >>> average_precision([1,3,5], [1,3,4,6], k=1) == 1.0
    >>> average_precision([1,3,5], [1,3,4,6], k=2) == 1.0
    >>> average_precision([1,3,5], [1,3,4,6], k=3) == 0.6666666666666666
    """
    if k is not None: recommendations = recommendations[:k]

    _sum = 0
    for r, i in zip(recommendations, range(1, len(recommendations) + 1)):
        if r in relevant_recommendations and r not in recommendations[:i-1]:  # r is relevant and the first occurrence
            _sum += precision(recommendations, relevant_recommendations, k=i)

    if k is None: return _sum / len(relevant_recommendations)
    return _sum / min(len(relevant_recommendations), k)


def mean_average_precision(recommendations_list, relevant_recommendations_list, k=None):
    """Mean Average Precision at k
    Example calls:
    >>> mean_average_precision([[0,1,1], [1,2,1]], [[1], [1,2]]) == 0.75
    >>> mean_average_precision([[1,3,5], [5,3,1]], [[1,3,4,6], [1,3,4,6]], k=1) == 0.5
    """
    _sum = 0
    for recommendations, relevant_recommendations in zip(recommendations_list, relevant_recommendations_list):
        _sum += average_precision(recommendations, relevant_recommendations, k=k)
    return _sum / len(recommendations_list)
