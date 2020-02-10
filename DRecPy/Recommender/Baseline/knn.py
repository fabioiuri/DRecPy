from abc import ABC
from abc import abstractmethod
from DRecPy.Recommender import RecommenderABC
from .similarity import cosine_sim, adjusted_cosine_sim
from scipy.sparse import csr_matrix
from heapq import nlargest


class BaseKNN(RecommenderABC, ABC):
    """Base Collaborative Filtering recommender abstract class.

    This class implements the skeleton methods for building a basic neighbour-based CF.
    The following methods are still required to be implemented:
        _fit(): should fit the model.
        _predict_neighbours(): should return a list of tuples (similarity, interaction).
        _predict_default(): should return the default prediction value that is used
            when a minimum number of neighbours is not found.

    Attributes:
        k: An integer representing the number of neighbours used to make a prediction.
            Default: 20.
        m: An integer representing the minimum number of co-rated users/items required
            to validate the similarity value (if not valid, sim. value is set to 0).
            Default: 5.
        sim_metric: Optional string representing the name of the similarity metric to use.
            Supported: 'adjusted_cosine', 'cosine'. Default: 'adjusted_cosine'.
        shrinkage: Optional integer representing the discounting factor for computing the
            similarity between items / users (discounts less when #co-ratings increases).
            Default: 100.
    """

    def __init__(self, k=20, m=5, sim_metric='adjusted_cosine', shrinkage=100, **kwds):
        super(BaseKNN, self).__init__(**kwds)
        self.sim_metric = sim_metric
        if sim_metric == 'adjusted_cosine':
            self.sim_metric_fn = adjusted_cosine_sim
        elif sim_metric == 'cosine':
            self.sim_metric_fn = adjusted_cosine_sim
        else:
            raise Exception('There is no similarity metric corresponding to the name "{}".'.format(name))
        self.k = k
        self.m = m
        self.interactions = None
        self.user_items = None
        self.similarities = None
        self.shrinkage = shrinkage

    def _do_batch(self, **kwds):
        raise NotImplementedError

    def _predict(self, uid, iid, **kwds):
        if uid is None and self.predicts_wo_user or iid is None and self.predicts_wo_item:
            return self._predict_default(uid, iid)

        neighbours = self._predict_neighbours(uid, iid)

        if len(neighbours) == 0: return self._predict_default(uid, iid)

        sim_sum = 1e-6
        interaction_sum = 0
        for neighbour in neighbours:
            similarity, interaction = neighbour
            interaction_sum += similarity * interaction
            sim_sum += similarity

        return interaction_sum / sim_sum

    @abstractmethod
    def _predict_neighbours(self, uid, iid):
        pass

    @abstractmethod
    def _predict_default(self, uid, iid):
        pass

    def _get_sim(self, id1, id2):
        """Computes the similarity between the provided items/users."""
        tmp = id1
        id1 = max(id1, id2)
        id2 = min(tmp, id2)
        if (id1, id2) not in self.similarities: return 0
        return self.similarities[id1, id2]


class ItemKNN(BaseKNN):
    """Item-based KNN Collaborative Filtering recommender model.

    Implementation of a basic item neighbour-based CF.

    Public Methods:
        fit(), predict(), recommend(), rank().

    Attributes: See parent object BaseKNN
    """

    def __init__(self, **kwds):
        super(ItemKNN, self).__init__(**kwds)

        self.predicts_wo_item = True

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # create storage data structures
        self.interactions = {}
        self.user_items = {}
        item_rated_users = {}
        interactions, uids, iids = [], [], []
        for record in self.interaction_dataset.values():
            u, i, r = record['uid'], record['iid'], record['interaction']
            interactions.append(r)
            uids.append(u)
            iids.append(i)

            if u not in self.interactions:
                self.interactions[u] = []
                self.user_items[u] = []
            if i not in item_rated_users:
                item_rated_users[i] = set()

            self.interactions[u].append(r)
            self.user_items[u].append(i)
            item_rated_users[i].add(u)

        # compute similarity matrix
        self._log('Computing similarity matrix...')
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (iids, uids)),
                                                            shape=(len(iids), len(uids)))).tocoo()
        self.similarities = {}
        for i1, i2, s in zip(similarities_matrix.row, similarities_matrix.col, similarities_matrix.data):
            if i1 <= i2:  # symmetric matrix - only need 1/2 of the values
                continue

            co_ratings = item_rated_users[i1] & item_rated_users[i2]
            if self.m > 0 and len(co_ratings) < self.m:  # not enough co-ratings
                continue

            self.similarities[(i1, i2)] = s

            if self.shrinkage is not None:
                self.similarities[(i1, i2)] *= len(co_ratings) / (len(co_ratings) + self.shrinkage + 1e-6)

    def _predict_neighbours(self, uid, iid):
        full_sim_list = [(self._get_sim(iid, iid2), r) for iid2, r in zip(self.user_items[uid], self.interactions[uid])]
        relev_sim_list = filter(lambda x: x[0] > 0, full_sim_list)
        return nlargest(self.k, relev_sim_list)

    def _predict_default(self, uid, _):
        """Returns the user average interaction."""
        return sum(self.interactions[uid]) / len(self.interactions[uid])


class UserKNN(BaseKNN):
    """User-based KNN Collaborative Filtering recommender model.

    Implementation of a basic user neighbour-based CF.

    Public Methods:
        fit(), predict(), recommend(), rank().

    Attributes: See parent object BaseKNN
    """

    def __init__(self, **kwds):
        super(UserKNN, self).__init__(**kwds)

        self.predicts_wo_user = True

        self.item_users = None

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # create storage data structures
        self.interactions = {}
        self.user_items = {}
        self.item_users = {}
        user_rated_items = {}
        interactions, uids, iids = [], [], []
        for record in self.interaction_dataset.values():
            u, i, r = record['uid'], record['iid'], record['interaction']
            interactions.append(r)
            uids.append(u)
            iids.append(i)

            if u not in self.user_items:
                self.user_items[u] = []
                user_rated_items[u] = set()
            if i not in self.interactions:
                self.interactions[i] = []
                self.item_users[i] = []

            self.interactions[i].append(r)
            self.item_users[i].append(u)
            self.user_items[u].append(i)
            user_rated_items[u].add(i)

        # compute similarity matrix
        self._log('Computing similarity matrix...')
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (uids, iids)),
                                                            shape=(len(uids), len(iids)))).tocoo()
        self.similarities = {}
        for u1, u2, s in zip(similarities_matrix.row, similarities_matrix.col, similarities_matrix.data):
            if u1 <= u2:  # symmetric matrix - only need 1/2 of the values
                continue

            co_ratings = user_rated_items[u1] & user_rated_items[u2]
            if self.m > 0 and len(co_ratings) < self.m:  # not enough co-ratings
                continue

            self.similarities[(u1, u2)] = s

            if self.shrinkage is not None:
                self.similarities[(u1, u2)] *= len(co_ratings) / (len(co_ratings) + self.shrinkage + 1e-6)

    def _predict_neighbours(self, uid, iid):
        full_sim_list = [(self._get_sim(uid, uid2), r) for uid2, r in zip(self.item_users[iid], self.interactions[iid])]
        relev_sim_list = filter(lambda x: x[0] > 0, full_sim_list)
        return nlargest(self.k, relev_sim_list)

    def _predict_default(self, _, iid):
        """Returns the item average interaction."""
        return sum(self.interactions[iid]) / len(self.interactions[iid])
