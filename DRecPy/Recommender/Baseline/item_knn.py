from .base_knn import BaseKNN
from scipy.sparse import csr_matrix
from heapq import nlargest


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

    def _compute_similarities(self):
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
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (iids, uids)))).tocoo()

        for i1, i2, s in zip(similarities_matrix.row, similarities_matrix.col, similarities_matrix.data):
            if i1 <= i2:  # symmetric matrix - only need 1/2 of the values
                continue

            co_ratings = item_rated_users[i1] & item_rated_users[i2]
            if self.m > 0 and len(co_ratings) < self.m:  # not enough co-ratings
                continue

            if i1 not in self._similarities: self._similarities[i1] = dict()
            self._similarities[i1][i2] = s

            if self.shrinkage is not None:
                self._similarities[i1][i2] *= len(co_ratings) / (len(co_ratings) + self.shrinkage + 1e-6)

    def _compute_neighbours(self):
        for iid in self.interaction_dataset.unique('iid').values('iid', to_list=True):
            self._neighbours[iid] = nlargest(self.k, filter(
                lambda x: x[0] is not None and x[0] > 0,
                [(self._get_sim(iid, iid2), iid2) for iid2 in range(self.n_items) if iid2 != iid]
            ))

    def _get_interaction(self, iid, uid):
        record = self.interaction_dataset.select_one(f'uid == {uid}, iid == {iid}')
        return None if record is None else record['interaction']

    def _predict_default(self, uid):
        """Returns the user average interaction."""
        user_records = self.interaction_dataset.select(f'uid == {uid}')
        return sum(user_records.values_list(columns=['interaction'], to_list=True)) / len(user_records)

    def _rank(self, uid, iids, n, novelty):
        iids = set(iids)
        if novelty:
            rated_items = self.interaction_dataset.select(f'uid == {uid}').values_list('iid', to_list=True)
            iids = iids.difference(set(rated_items))

        pred_list = []
        user_ds = self.interaction_dataset.select(f'uid == {uid}')
        user_iid_interactions = set(user_ds.values_list('iid', to_list=True))

        for iid in iids:
            interactions, similarities = [], []
            for similarity, neighbour in self._neighbours[iid]:
                if neighbour not in user_iid_interactions: continue
                interaction = user_ds.select_one(f'iid == {neighbour}', 'interaction', to_list=True)
                if interaction is None: continue
                interactions.append(interaction)
                similarities.append(similarity)

            if len(interactions) == 0 and self.use_averages:
                pred = self._predict_default(uid)
            else:
                pred = self.aggregation_fn(interactions, similarities)

            if pred is not None:
                pred_list.append((pred, iid))

        return nlargest(n, pred_list)