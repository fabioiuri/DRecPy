from DRecPy.Evaluation.Metrics import DCG
from DRecPy.Evaluation.Metrics import NDCG
from DRecPy.Evaluation.Metrics import HitRatio
from DRecPy.Evaluation.Metrics import ReciprocalRank
from DRecPy.Evaluation.Metrics import Recall
from DRecPy.Evaluation.Metrics import Precision
from DRecPy.Evaluation.Metrics import FScore
from DRecPy.Evaluation.Metrics import AveragePrecision


def test_dcg_0():
    assert round(DCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 45.64


def test_dcg_1():
    assert round(DCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 10.27


def test_dcg_2():
    assert DCG(strong_relevancy=True)([6], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}) == 3.0


def test_dcg_3():
    assert DCG(strong_relevancy=False)([6], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}) == 2.0


def test_dcg_4():
    assert round(DCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, k=2), 2) == 40.46


def test_dcg_5():
    assert round(DCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, k=2), 2) == 7.52


def test_ndcg_0():
    assert NDCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}) == 1.0


def test_ndcg_1():
    assert NDCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}) == 1.0


def test_ndcg_2():
    assert round(NDCG(strong_relevancy=True)([6], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 0.07


def test_ndcg_3():
    assert round(NDCG(strong_relevancy=False)([6], relevancies={1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 0.19


def test_ndcg_5():
    assert round(NDCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 0.69


def test_ndcg_6():
    assert round(NDCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}), 2) == 0.72


def test_ndcg_7():
    assert round(NDCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}), 2) == 0.56


def test_ndcg_8():
    assert round(NDCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}), 2) == 0.69


def test_ndcg_9():
    assert round(NDCG(strong_relevancy=True)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, k=3), 2) == 0.29


def test_ndcg_10():
    assert round(NDCG(strong_relevancy=False)([1, 3, 2, 6, 5, 4], relevancies={1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, k=3), 2) == 0.45


def test_hit_ratio_0():
    assert HitRatio()([1, 2, 3], relevant_recommendations=[3]) == 1.0


def test_hit_ratio_1():
    assert HitRatio()([1, 2, 3], relevant_recommendations=[3, 2]) == 1.0


def test_hit_ratio_2():
    assert round(HitRatio()([1, 2, 3], relevant_recommendations=[3, 2, 5]), 2) == 0.67


def test_hit_ratio_3():
    assert HitRatio()([1, 2, 3], relevant_recommendations=[3, 2], k=2) == 0.5


def test_hit_ratio_4():
    assert round(HitRatio()([1, 2], relevant_recommendations=[3, 2, 5]), 2) == 0.33


def test_hit_ratio_5():
    assert round(HitRatio()([1, 2], relevant_recommendations=[3, 2, 5], k=2), 2) == 0.33


def test_hit_ratio_6():
    assert HitRatio()([1, 2], relevant_recommendations=[3, 2, 5], k=1) == 0


def test_reciprocal_rank_0():
    assert ReciprocalRank()([1, 2, 3], relevant_recommendation=1) == 1.0


def test_reciprocal_rank_1():
    assert ReciprocalRank()([1, 2, 3], relevant_recommendation=2) == 0.5


def test_reciprocal_rank_2():
    assert round(ReciprocalRank()([1, 2, 3], relevant_recommendation=3), 2) == 0.33


def test_reciprocal_rank_3():
    assert ReciprocalRank()([1, 2, 3], relevant_recommendation=4) == 0


def test_reciprocal_rank_4():
    assert ReciprocalRank()([1, 2, 3], relevant_recommendation=3, k=2) == 0


def test_reciprocal_rank_5():
    assert ReciprocalRank()([1, 2, 3], relevant_recommendation=2, k=2) == 0.5


def test_recall_0():
    assert Recall()([1, 2, 3], relevant_recommendations=[1, 2, 3]) == 1.0


def test_recall_1():
    assert Recall()([1, 2, 3], relevant_recommendations=[2, 3]) == 1.0


def test_recall_2():
    assert round(Recall()([1, 2, 3], relevant_recommendations=[2, 3, 4]), 2) == 0.67


def test_recall_3():
    assert Recall()([1, 2, 3], relevant_recommendations=[4]) == 0.0


def test_recall_4():
    assert Recall()([1, 2, 3], relevant_recommendations=[2, 3], k=2) == 0.5


def test_precision_0():
    assert Precision()([1, 2, 3], relevant_recommendations=[1, 2, 3]) == 1.0


def test_precision_1():
    assert round(Precision()([1, 2, 3], relevant_recommendations=[2, 3]), 2) == 0.67


def test_precision_2():
    assert round(Precision()([1, 2, 3], relevant_recommendations=[2, 3, 4]), 2) == 0.67


def test_precision_3():
    assert Precision()([1, 2, 3], relevant_recommendations=[4]) == 0.0


def test_precision_4():
    assert Precision()([1, 2, 3], relevant_recommendations=[2, 3], k=2) == 0.5


def test_f_score_0():
    assert FScore()([1], relevant_recommendations=[1, 2, 4, 8]) == 0.4


def test_f_score_1():
    assert round(FScore()([1, 2], relevant_recommendations=[1, 2, 4, 8]), 2) == 0.67


def test_f_score_2():
    assert round(FScore()([1, 2, 3], relevant_recommendations=[1, 2, 4, 8]), 2) == 0.57


def test_f_score_3():
    assert FScore()([1, 2, 3, 4], relevant_recommendations=[1, 2, 4, 8]) == 0.75


def test_f_score_4():
    assert round(FScore()([1, 2, 3, 4, 5, 6, 7, 8], relevant_recommendations=[1, 2, 4, 8]), 2) == 0.67


def test_f_score_5():
    assert FScore()([1, 2, 3, 4, 5, 6, 7, 8], relevant_recommendations=[1, 2, 4, 8], k=4) == 0.75


def test_f_score_6():
    assert FScore()([1, 2, 4, 8], relevant_recommendations=[1, 2, 4, 8]) == 1.0


def test_f_score_7():
    assert round(FScore(beta=2)([1], relevant_recommendations=[1, 2, 4, 8]), 2) == 0.29


def test_average_precision_0():
    assert AveragePrecision()([0, 1, 1], relevant_recommendations=[1]) == 0.5


def test_average_precision_1():
    assert AveragePrecision()([0, 1, 0], relevant_recommendations=[1]) == 0.5


def test_average_precision_2():
    assert round(AveragePrecision()([0, 0, 1], relevant_recommendations=[1]), 2) == 0.33


def test_average_precision_3():
    assert AveragePrecision()([0, 0, 0], relevant_recommendations=[1]) == 0.0


def test_average_precision_4():
    assert AveragePrecision()([1, 0, 0], relevant_recommendations=[1]) == 1.0


def test_average_precision_5():
    assert AveragePrecision()([1, 2, 1], relevant_recommendations=[1, 2]) == 1.0


def test_average_precision_6():
    assert AveragePrecision()([2, 1, 1], relevant_recommendations=[1, 2]) == 1


def test_average_precision_8():
    assert AveragePrecision()([1, 3, 5], relevant_recommendations=[1, 3, 4, 6]) == 0.5


def test_average_precision_9():
    assert AveragePrecision()([1, 3, 5], relevant_recommendations=[1, 3, 4, 6], k=1) == 1.0


def test_average_precision_10():
    assert AveragePrecision()([1, 3, 5], relevant_recommendations=[1, 3, 4, 6], k=2) == 1.0


def test_average_precision_11():
    assert round(AveragePrecision()([1, 3, 5], relevant_recommendations=[1, 3, 4, 6], k=3), 2) == 0.67
