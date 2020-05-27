from DRecPy.Evaluation.Metrics import dcg
from DRecPy.Evaluation.Metrics import ndcg
from DRecPy.Evaluation.Metrics import hit_ratio
from DRecPy.Evaluation.Metrics import reciprocal_rank
from DRecPy.Evaluation.Metrics import recall
from DRecPy.Evaluation.Metrics import precision
from DRecPy.Evaluation.Metrics import f_score
from DRecPy.Evaluation.Metrics import average_precision


def test_dcg_0():
    assert round(dcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True), 2) == 45.64


def test_dcg_1():
    assert round(dcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False), 2) == 10.27


def test_dcg_2():
    assert dcg([6], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True) == 3.0


def test_dcg_3():
    assert dcg([6], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False) == 2.0


def test_dcg_4():
    assert round(dcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True, k=2), 2) == 40.46


def test_dcg_5():
    assert round(dcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False, k=2), 2) == 7.52


def test_ndcg_0():
    assert ndcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True) == 1.0


def test_ndcg_1():
    assert ndcg([1, 3, 2, 6, 5, 4], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False) == 1.0


def test_ndcg_2():
    assert round(ndcg([6], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True), 2) == 0.07


def test_ndcg_3():
    assert round(ndcg([6], {1: 5, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False), 2) == 0.19


def test_ndcg_5():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=True), 2) == 0.69


def test_ndcg_6():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2}, strong_relevancy=False), 2) == 0.72


def test_ndcg_7():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, strong_relevancy=True), 2) == 0.56


def test_ndcg_8():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, strong_relevancy=False), 2) == 0.69


def test_ndcg_9():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, strong_relevancy=True, k=3), 2) == 0.29


def test_ndcg_10():
    assert round(ndcg([1, 3, 2, 6, 5, 4], {1: 0, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2}, strong_relevancy=False, k=3), 2) == 0.45


def test_hit_ratio_0():
    assert hit_ratio([1, 2, 3], [3]) == 1.0


def test_hit_ratio_1():
    assert hit_ratio([1, 2, 3], [3, 2]) == 1.0


def test_hit_ratio_2():
    assert round(hit_ratio([1, 2, 3], [3, 2, 5]), 2) == 0.67


def test_hit_ratio_3():
    assert hit_ratio([1, 2, 3], [3, 2], k=2) == 0.5


def test_hit_ratio_4():
    assert round(hit_ratio([1, 2], [3, 2, 5]), 2) == 0.33


def test_hit_ratio_5():
    assert round(hit_ratio([1, 2], [3, 2, 5], k=2), 2) == 0.33


def test_hit_ratio_6():
    assert hit_ratio([1, 2], [3, 2, 5], k=1) == 0


def test_reciprocal_rank_0():
    assert reciprocal_rank([1, 2, 3], 1) == 1.0


def test_reciprocal_rank_1():
    assert reciprocal_rank([1, 2, 3], 2) == 0.5


def test_reciprocal_rank_2():
    assert round(reciprocal_rank([1, 2, 3], 3), 2) == 0.33


def test_reciprocal_rank_3():
    assert reciprocal_rank([1, 2, 3], 4) == 0


def test_reciprocal_rank_4():
    assert reciprocal_rank([1, 2, 3], 3, k=2) == 0


def test_reciprocal_rank_5():
    assert reciprocal_rank([1, 2, 3], 2, k=2) == 0.5


def test_recall_0():
    assert recall([1, 2, 3], [1, 2, 3]) == 1.0


def test_recall_1():
    assert recall([1, 2, 3], [2, 3]) == 1.0


def test_recall_2():
    assert round(recall([1, 2, 3], [2, 3, 4]), 2) == 0.67


def test_recall_3():
    assert recall([1, 2, 3], [4]) == 0.0


def test_recall_4():
    assert recall([1, 2, 3], [2, 3], k=2) == 0.5


def test_precision_0():
    assert precision([1, 2, 3], [1, 2, 3]) == 1.0


def test_precision_1():
    assert round(precision([1, 2, 3], [2, 3]), 2) == 0.67


def test_precision_2():
    assert round(precision([1, 2, 3], [2, 3, 4]), 2) == 0.67


def test_precision_3():
    assert precision([1, 2, 3], [4]) == 0.0


def test_precision_4():
    assert precision([1, 2, 3], [2, 3], k=2) == 0.5


def test_f_score_0():
    assert f_score([1], [1, 2, 4, 8]) == 0.4


def test_f_score_1():
    assert round(f_score([1, 2], [1, 2, 4, 8]), 2) == 0.67


def test_f_score_2():
    assert round(f_score([1, 2, 3], [1, 2, 4, 8]), 2) == 0.57


def test_f_score_3():
    assert f_score([1, 2, 3, 4], [1, 2, 4, 8]) == 0.75


def test_f_score_4():
    assert round(f_score([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 8]), 2) == 0.67


def test_f_score_5():
    assert f_score([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 4, 8], k=4) == 0.75


def test_f_score_6():
    assert f_score([1, 2, 4, 8], [1, 2, 4, 8]) == 1.0


def test_f_score_7():
    assert round(f_score([1], [1, 2, 4, 8], beta=2), 2) == 0.29


def test_average_precision_0():
    assert average_precision([0, 1, 1], [1]) == 0.5


def test_average_precision_1():
    assert average_precision([0, 1, 0], [1]) == 0.5


def test_average_precision_2():
    assert round(average_precision([0, 0, 1], [1]), 2) == 0.33


def test_average_precision_3():
    assert average_precision([0, 0, 0], [1]) == 0.0


def test_average_precision_4():
    assert average_precision([1, 0, 0], [1]) == 1.0


def test_average_precision_5():
    assert average_precision([1, 2, 1], [1, 2]) == 1.0


def test_average_precision_6():
    assert average_precision([2, 1, 1], [1, 2]) == 1


def test_average_precision_8():
    assert average_precision([1, 3, 5], [1, 3, 4, 6]) == 0.5


def test_average_precision_9():
    assert average_precision([1, 3, 5], [1, 3, 4, 6], k=1) == 1.0


def test_average_precision_10():
    assert average_precision([1, 3, 5], [1, 3, 4, 6], k=2) == 1.0


def test_average_precision_11():
    assert round(average_precision([1, 3, 5], [1, 3, 4, 6], k=3), 2) == 0.67
