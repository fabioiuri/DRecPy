import math
from abc import abstractmethod
from .metric_abc import MetricABC


class RankingMetricABC(MetricABC):
    @abstractmethod
    def __call__(self, recommendations, k=None):
        """
         Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None (no truncation occurs).
        Returns:
            The computed metric value.
        """
        pass


class DCG(RankingMetricABC):
    """Discounted Cumulative Gain at k.

    Args:
        strong_relevancy: An optional boolean indicating which variant of the DCG is used. If set to True,
            usually results in smaller values than when it's set to False. Default: True.
    """
    def __init__(self, strong_relevancy=True):
        super(DCG, self).__init__()

        self.strong_relevancy = strong_relevancy

    def __call__(self, recommendations, k=None, relevancies=None):
        """Computes the Discounted Cumulative Gain at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevancies: A dict that should map recommended ids to their relevancy value.

        Returns:
            The computed Discounted Cumulative Gain value.
        """
        if relevancies is None: return 0
        if k is not None: recommendations = recommendations[:k]

        curr_dcg = 0
        for i, r in enumerate(recommendations):
            rel = float(relevancies[r])

            if self.strong_relevancy:
                curr_dcg += (2 ** rel - 1) / math.log2(2 + i)
            else:
                curr_dcg += rel / math.log2(2 + i)

        return curr_dcg


class NDCG(RankingMetricABC):
    """Discounted Cumulative Gain at k.

    Args:
        strong_relevancy: An optional boolean indicating which variant of the DCG is used. If set to True,
            usually results in smaller values than when it's set to False. Default: True.
    """
    def __init__(self, strong_relevancy=True):
        super(NDCG, self).__init__()

        self.strong_relevancy = strong_relevancy
        self.dcg = DCG(strong_relevancy=strong_relevancy)

    def __call__(self, recommendations, k=None, relevancies=None):
        """Computes the Normalized Discounted Cumulative Gain at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevancies: A dict that should map ids to their relevancy value. Note that both relevant and irrelevant
                item identifiers should be present on this dict, so that the ideal recommendation list can be computed.

        Returns:
            The computed Normalized Discounted Cumulative Gain value.
        """
        if relevancies is None: return 0

        curr_dcg = self.dcg(recommendations, relevancies=relevancies, k=k)
        best_recommendations = sorted(relevancies.keys(), key=lambda x: -relevancies[x])
        best_dcg = self.dcg(best_recommendations, relevancies=relevancies, k=k)

        return curr_dcg / best_dcg


class HitRatio(RankingMetricABC):
    """Hit Ratio at k."""
    def __call__(self, recommendations, k=None, relevant_recommendations=None):
        """Computes the Hit Ratio at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendations: A list with the identifiers of the relevant recommendation items.

        Returns:
            The computed Hit Ratio value.
        """
        if relevant_recommendations is None: return 0
        if k is not None: recommendations = recommendations[:k]

        recommendations = set([str(item) for item in recommendations])
        relevant_recommendations = set([str(item) for item in relevant_recommendations])

        return len(recommendations.intersection(relevant_recommendations)) / len(relevant_recommendations)


class ReciprocalRank(RankingMetricABC):
    """Reciprocal Rank at k."""
    def __call__(self, recommendations, k=None, relevant_recommendation=None):
        """Computes the Reciprocal Rank at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendation: The identifier of the most relevant item.

        Returns:
            The computed Reciprocal Rank value.
        """
        if relevant_recommendation is None: return 0
        if k is not None: recommendations = recommendations[:k]

        if relevant_recommendation in recommendations:
            return 1 / (recommendations.index(relevant_recommendation) + 1)
        return 0


class Recall(RankingMetricABC):
    """Recall at k."""
    def __call__(self, recommendations, k=None, relevant_recommendations=None):
        """Computes the Recall at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendations: A list with the identifiers of the relevant recommendation items.

        Returns:
            The computed Recall value.
        """
        if relevant_recommendations is None: return 0
        if k is not None: recommendations = recommendations[:k]

        in_common = set(recommendations).intersection(set(relevant_recommendations))
        return len(in_common) / len(relevant_recommendations)


class Precision(RankingMetricABC):
    """Precision at k."""
    def __call__(self, recommendations, k=None, relevant_recommendations=None):
        """Computes the Precision at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendations: A list with the identifiers of the relevant recommendation items.

        Returns:
            The computed Precision value.
        """
        if relevant_recommendations is None: return 0
        if k is not None: recommendations = recommendations[:k]

        in_common = set(recommendations).intersection(set(relevant_recommendations))
        return len(in_common) / len(recommendations)


class FScore(RankingMetricABC):
    """F-score at k.

    Args:
        beta: An optional integer representing the weight of the recall value on the combined score.
            Beta > 1 favours recall over precision, while beta < 1 favours precision over recall. Default: 1.
    """
    def __init__(self, beta=1):
        super(FScore, self).__init__()

        self.beta = beta
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self, recommendations, k=None, relevant_recommendations=None):
        """Computes the F-score at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendations: A list with the identifiers of the relevant recommendation items.

        Returns:
            The computed F-score value.
        """
        if relevant_recommendations is None: return 0

        p = self.precision(recommendations, relevant_recommendations=relevant_recommendations, k=k)
        r = self.recall(recommendations, relevant_recommendations=relevant_recommendations, k=k)

        return (1 + self.beta ** 2) * p * r / ((self.beta ** 2 * p) + r)


class AveragePrecision(RankingMetricABC):
    """Average Precision at k."""

    def __init__(self):
        super(AveragePrecision, self).__init__()

        self.precision = Precision()

    def __call__(self, recommendations, k=None, relevant_recommendations=None):
        """Computes the Average Precision at k.

        Args:
            recommendations: A list with the identifiers of the recommended items.
            k: Optional integer denoting the truncation parameter. If defined, the recommendations list is
                truncated after the k-th element. Default: None.
            relevant_recommendations: A list with the identifiers of the relevant recommendation items.

        Returns:
            The computed Average Precision value.
        """
        if relevant_recommendations is None: return 0
        if k is not None: recommendations = recommendations[:k]

        _sum = 0
        for r, i in zip(recommendations, range(1, len(recommendations) + 1)):
            if r in relevant_recommendations and r not in recommendations[
                                                          :i - 1]:  # r is relevant and the first occurrence
                _sum += self.precision(recommendations, relevant_recommendations=relevant_recommendations, k=i)

        if k is None: return _sum / len(relevant_recommendations)
        return _sum / min(len(relevant_recommendations), k)
