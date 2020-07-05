from DRecPy.Recommender import RecommenderABC


class TestRecommender(RecommenderABC):

    def __init__(self, **kwds):
        super(TestRecommender, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # used to declare variables and the neural network structure of the model
        print(f'doing pre-fit with learning_rate={learning_rate}, neg_ratio={neg_ratio}, reg_rate={reg_rate}')
        pass

    def _do_batch(self, batch_size, **kwds):
        # OPTIONAL METHOD that calls for each epoch with the batch_size and other custom arguments
        # if your model trains in one go, add that logic to the _pre_fit method and raise a NotImplementedError here
        print('doing batch of size', batch_size)
        return 0  # loss value obtained on the current batch

    def _predict(self, uid, iid, **kwds):
        return 5  # predict for a (user, item) pair


class TestRecommenderNonDeepLearning(RecommenderABC):

    def __init__(self, **kwds):
        super(TestRecommenderNonDeepLearning, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # used to declare variables and the neural network structure of the model
        print(f'doing pre-fit with learning_rate={learning_rate}, neg_ratio={neg_ratio}, reg_rate={reg_rate}')
        pass

    def _do_batch(self, batch_size, **kwds):
        # OPTIONAL METHOD that calls for each epoch with the batch_size and other custom arguments
        # if your model trains in one go, add that logic to the _pre_fit method and raise a NotImplementedError here
        raise NotImplementedError

    def _predict(self, uid, iid, **kwds):
        return 5  # predict for a (user, item) pair


# Use the new recommender
from DRecPy.Dataset import get_train_dataset

ds_train = get_train_dataset('ml-100k', verbose=False)

print('TestRecommender')
recommender = TestRecommender(verbose=False)
recommender.fit(ds_train, epochs=2, batch_size=10)
print(recommender.predict(1, 1))

print('TestRecommenderNonDeepLearning')
recommender = TestRecommenderNonDeepLearning(verbose=False)
recommender.fit(ds_train, epochs=2, batch_size=10)
print(recommender.predict(1, 1))
