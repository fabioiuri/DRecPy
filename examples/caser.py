from DRecPy.Recommender import Caser
from DRecPy.Dataset import get_full_dataset
from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Evaluation.Processes import recommendation_evaluation
from DRecPy.Evaluation.Metrics import Precision
from DRecPy.Evaluation.Metrics import Recall
from DRecPy.Evaluation.Metrics import AveragePrecision

ds = get_full_dataset('ml-1m')
ds.apply('interaction', lambda x: 1 if x > 0 else 0)
ds_train, ds_test = leave_k_out(ds, k=0.2, last_timestamps=True, seed=0)

caser = Caser(L=5, T=3, d=50, n_v=4, n_h=16, dropout_rate=0.5, sort_column='timestamp', seed=10)
caser.fit(ds_train, epochs=350, batch_size=2 ** 12, learning_rate=0.005, reg_rate=1e-6, neg_ratio=3)


print(recommendation_evaluation(caser, ds_test, novelty=True, k=[1, 5, 10],
                                metrics=[AveragePrecision(), Precision(), Recall()], seed=10))

# 'AveragePrecision@1': 0.232, 'AveragePrecision@5': 0.1378, 'AveragePrecision@10': 0.1123,
# 'Precision@1': 0.232, 'Precision@5': 0.2088, 'Precision@10': 0.1899,
# 'Recall@1': 0.0138, 'Recall@5': 0.062, 'Recall@10': 0.1085
