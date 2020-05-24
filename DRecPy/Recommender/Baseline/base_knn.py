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


class BaseKNN(RecommenderABC, ABC):
    """Base Collaborative Filtering recommender abstract class.

    This class implements the skeleton methods for building a basic neighbour-based CF.
    The following methods are still required to be implemented:
    _fit(): should fit the model.
    _predict_default(): should return the default prediction value that is used
    when a minimum number of neighbours is not found. Only used when use_averages=True.

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
                 use_averages=False, **kwds):
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

        self._similarities = dict()
        self._neighbours = dict()

    def _pre_fit(self, learning_rate, neg_ratio, reg_rate, **kwds):
        self._log('Computing similarity matrix...')
        self._compute_similarities()

        self._log('Computing neighbours...')
        self._compute_neighbours()

    def _do_batch(self, **kwds):
        raise NotImplementedError

    def _predict(self, uid, iid, **kwds):
        if uid is None or iid is None: return None

        eligible_neighbours, interactions, similarities = [], [], []
        for similarity, neighbour in self._neighbours[uid if self.type == 'user' else iid]:
            interaction = self._get_interaction(neighbour, iid if self.type == 'user' else uid)
            if interaction is None: continue
            eligible_neighbours.append(neighbour)
            interactions.append(interaction)
            similarities.append(similarity)

        if len(eligible_neighbours) == 0 and self.use_averages:
            return self._predict_default(iid if self.type == 'user' else uid)

        return self.aggregation_fn(eligible_neighbours, interactions, similarities)

    @abstractmethod
    def _predict_default(self, _):
        pass

    @abstractmethod
    def _get_interaction(self, _, __):
        pass

    @abstractmethod
    def _compute_similarities(self):
        pass

    @abstractmethod
    def _compute_neighbours(self):
        pass

    def _get_sim(self, id1, id2):
        """Computes the similarity between the provided items/users."""
        tmp = id1
        id1 = max(id1, id2)
        id2 = min(tmp, id2)
        if id1 not in self._similarities or id2 not in self._similarities[id1]: return 0
        return self._similarities[id1][id2]
