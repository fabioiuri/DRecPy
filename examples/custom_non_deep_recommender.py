from DRecPy.Recommender import RecommenderABC
from DRecPy.Dataset import get_train_dataset


class TestRecommenderNonDeepLearning(RecommenderABC):

    def __init__(self, **kwds):
        super(TestRecommenderNonDeepLearning, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # used to declare variables and do the non-deep learning fit process, such as computing similarities and
        # neighbours for knn-based models
        self._info(f'doing pre-fit with learning_rate={learning_rate}, neg_ratio={neg_ratio}, reg_rate={reg_rate}')
        pass

    def _sample_batch(self, batch_size, **kwds):
        raise NotImplemented  # since it's non-deep learning based, no need for batch training

    def _predict_batch(self, batch_samples, **kwds):
        raise NotImplemented  # since it's non-deep learning based, no need for batch training

    def _compute_batch_loss(self, predictions, desired_values, **kwds):
        raise NotImplemented  # since it's non-deep learning based, no need for batch training

    def _compute_reg_loss(self, reg_rate, batch_size, **kwds):
        raise NotImplemented  # since it's non-deep learning based, no need for batch training

    def _predict(self, uid, iid, **kwds):
        return 5  # predict for a (user, item) pair


ds_train = get_train_dataset('ml-100k', verbose=False)

print('TestRecommenderNonDeepLearning')
recommender = TestRecommenderNonDeepLearning(verbose=True)
recommender.fit(ds_train, epochs=2, batch_size=10)
print(recommender.predict(1, 1))
