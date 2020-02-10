from DRecPy.Recommender import RecommenderABC


class TestRecommender(RecommenderABC):

    def __init__(self, **kwds):
        super(TestRecommender, self).__init__(**kwds)

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # used to declare variables and the neural network structure of the model
        pass

    def _do_batch(self, **kwds):
        # each batch
        return 0  # loss value obtained on the current batch

    def _predict(self, uid, iid, **kwds):
        return 5  # predict for a (user, item) pair


# Use the new recommender
from DRecPy.Dataset import get_train_dataset

ds_train = get_train_dataset('ml-100k')

recommender = TestRecommender()
recommender.fit(ds_train)
print(recommender.predict(1, 1))
