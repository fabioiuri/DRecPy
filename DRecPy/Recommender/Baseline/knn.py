from abc import ABC
from abc import abstractmethod
from DRecPy.Recommender import RecommenderABC
from .similarity import cosine_sim
from .similarity import adjusted_cosine_sim
from .similarity import cosine_sim_cf
from .similarity import jaccard_sim
from .similarity import msd
from .similarity import pearson_corr
from .aggregation import mean
from .aggregation import weighted_mean
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
            Supported: 'adjusted_cosine', 'cosine', 'cosine_cf', 'jaccard', 'msd', 'pearson'. Default: 'adjusted_cosine'.
        aggregation: Optional string representing the name of the aggregation approach to use.
            Supported: 'mean', 'weighted_mean'. Default: 'weighted_mean'.
        shrinkage: Optional integer representing the discounting factor for computing the
            similarity between items / users (discounts less when #co-ratings increases).
            Default: 100.
        use_averages: Optional boolean indicating whether to use item (for UserKNN) or user
            (for ItemKNN) averages when no neighbours are found. Default: True.
    """

    def __init__(self, k=20, m=5, sim_metric='adjusted_cosine', aggregation='weighted_mean', shrinkage=100,
                 use_averages=True, **kwds):
        super(BaseKNN, self).__init__(**kwds)

        if sim_metric == 'adjusted_cosine':
            self.sim_metric_fn = adjusted_cosine_sim
        elif sim_metric == 'cosine':
            self.sim_metric_fn = cosine_sim
        elif sim_metric == 'cosine_cf':
            self.sim_metric_fn = cosine_sim_cf
        elif sim_metric == 'jaccard':
            self.sim_metric_fn = jaccard_sim
        elif sim_metric == 'msd':
            self.sim_metric_fn = msd
        elif sim_metric == 'pearson':
            self.sim_metric_fn = pearson_corr
        else:
            raise Exception(f'There is no similarity metric corresponding to the name "{sim_metric}".')

        if aggregation == 'mean':
            self.aggregation_fn = mean
        elif aggregation == 'weighted_mean':
            self.aggregation_fn = weighted_mean
        else:
            raise Exception(f'There is no aggregation approach corresponding to the name "{aggregation}".')

        self.k = k
        self.m = m
        self.type = None
        self.shrinkage = shrinkage
        self.use_averages = use_averages

        self.similarities = None
        self._neighbours_cache = dict()

    def _do_batch(self, **kwds):
        raise NotImplementedError

    def _predict(self, uid, iid, **kwds):
        if uid is None or iid is None: return None

        eligible_neighbours, interactions, similarities = [], [], []
        for similarity, neighbour in self._predict_neighbours(uid if self.type == 'user' else iid):
            interaction = self._get_interaction(neighbour, iid if self.type == 'user' else uid)
            if interaction is None: continue
            eligible_neighbours.append(neighbour)
            interactions.append(interaction)
            similarities.append(similarity)

        if len(eligible_neighbours) == 0 and self.use_averages:
            return self._predict_default(iid if self.type == 'user' else uid)

        return self.aggregation_fn(eligible_neighbours, interactions, similarities)

    @abstractmethod
    def _predict_neighbours(self, _):
        pass

    @abstractmethod
    def _predict_default(self, _):
        pass

    @abstractmethod
    def _get_interaction(self, _, __):
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

    Attributes: See parent object BaseKNN obj:`DRecPy.Recommender.Baseline.BaseKNN`
    """

    def __init__(self, **kwds):
        super(ItemKNN, self).__init__(**kwds)
        self.type = 'item'

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # create storage data structures
        item_rated_users = {}
        interactions, uids, iids = [], [], []
        for record in self.interaction_dataset.values():
            u, i, r = record['uid'], record['iid'], record['interaction']
            interactions.append(r)
            uids.append(u)
            iids.append(i)

            if i not in item_rated_users:
                item_rated_users[i] = set()

            item_rated_users[i].add(u)

        # compute similarity matrix
        self._log('Computing similarity matrix...')
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (iids, uids)))).tocoo()

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

    def _get_interaction(self, iid, uid):
        record = self.interaction_dataset.select_one(f'uid == {uid}, iid == {iid}')
        return None if record is None else record['interaction']

    def _predict_neighbours(self, iid):
        if iid in self._neighbours_cache: return self._neighbours_cache[iid]

        full_sim_list = [(self._get_sim(iid, iid2), iid2) for iid2 in range(self.n_items) if iid2 != iid]
        relev_sim_list = filter(lambda x: x[0] > 0, full_sim_list)
        neighbours = nlargest(self.k, relev_sim_list)

        try: self._neighbours_cache[iid] = neighbours
        except MemoryError: pass

        return self._neighbours_cache[iid]

    def _predict_default(self, uid):
        """Returns the user average interaction."""
        user_records = self.interaction_dataset.select(f'uid == {uid}')
        return sum(user_records.values_list(columns=['interaction'], to_list=True)) / len(user_records)


class UserKNN(BaseKNN):
    """User-based KNN Collaborative Filtering recommender model.

    Implementation of a basic user neighbour-based CF.

    Public Methods:
        fit(), predict(), recommend(), rank().

    Attributes: See parent object BaseKNN
    """

    def __init__(self, **kwds):
        super(UserKNN, self).__init__(**kwds)
        self.type = 'user'

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        # create storage data structures
        user_rated_items = {}
        interactions, uids, iids = [], [], []
        for record in self.interaction_dataset.values():
            u, i, r = record['uid'], record['iid'], record['interaction']
            interactions.append(r)
            uids.append(u)
            iids.append(i)

            if u not in user_rated_items:
                user_rated_items[u] = set()

            user_rated_items[u].add(i)

        # compute similarity matrix
        self._log('Computing similarity matrix...')
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (uids, iids)))).tocoo()
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

    def _get_interaction(self, uid, iid):
        record = self.interaction_dataset.select_one(f'uid == {uid}, iid == {iid}')
        return None if record is None else record['interaction']

    def _predict_neighbours(self, uid):
        if uid in self._neighbours_cache: return self._neighbours_cache[uid]

        full_sim_list = [(self._get_sim(uid, uid2), uid2) for uid2 in range(self.n_users) if uid2 != uid]
        relev_sim_list = filter(lambda x: x[0] > 0, full_sim_list)
        neighbours = nlargest(self.k, relev_sim_list)

        try: self._neighbours_cache[uid] = neighbours
        except MemoryError: pass

        return neighbours

    def _predict_default(self, iid):
        """Returns the item average interaction."""
        item_records = self.interaction_dataset.select(f'iid == {iid}')
        return sum(item_records.values_list(columns=['interaction'], to_list=True)) / len(item_records)
