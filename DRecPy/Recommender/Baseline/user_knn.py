from .base_knn import BaseKNN
from scipy.sparse import csr_matrix
from heapq import nlargest


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

    def _compute_similarities(self):
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
        similarities_matrix = self.sim_metric_fn(csr_matrix((interactions, (uids, iids)))).tocoo()

        for u1, u2, s in zip(similarities_matrix.row, similarities_matrix.col, similarities_matrix.data):
            if u1 <= u2:  # symmetric matrix - only need 1/2 of the values
                continue

            co_ratings = user_rated_items[u1] & user_rated_items[u2]
            if self.m > 0 and len(co_ratings) < self.m:  # not enough co-ratings
                continue

            if u1 not in self._similarities: self._similarities[u1] = dict()
            self._similarities[u1][u2] = s

            if self.shrinkage is not None:
                self._similarities[u1][u2] *= len(co_ratings) / (len(co_ratings) + self.shrinkage + 1e-6)

    def _compute_neighbours(self):
        for uid in self.interaction_dataset.unique('uid').values('uid', to_list=True):
            self._neighbours[uid] = nlargest(self.k, filter(
                lambda x: x[0] is not None and x[0] > 0,
                [(self._get_sim(uid, uid2), uid2) for uid2 in range(self.n_users) if uid2 != uid]
            ))

    def _get_interaction(self, uid, iid):
        record = self.interaction_dataset.select_one(f'uid == {uid}, iid == {iid}')
        return None if record is None else record['interaction']

    def _predict_default(self, iid):
        """Returns the item average interaction."""
        item_records = self.interaction_dataset.select(f'iid == {iid}')
        return sum(item_records.values_list(columns=['interaction'], to_list=True)) / len(item_records)

    def _rank(self, uid, iids, n, novelty):
        iids = set(iids)
        if novelty:
            rated_items = self.interaction_dataset.select(f'uid == {uid}').values_list('iid', to_list=True)
            iids = iids.difference(set(rated_items))

        tmp_iid_info = dict()

        for similarity, neighbour in self._neighbours[uid]:
            neighbour_records = self.interaction_dataset.select(f'uid == {neighbour}')
            for iid, interaction in neighbour_records.values(['iid', 'interaction'], to_list=True):
                if iid not in iids: continue
                if iid not in tmp_iid_info: tmp_iid_info[iid] = ([], [])
                tmp_iid_info[iid][0].append(interaction)
                tmp_iid_info[iid][1].append(similarity)

        pred_list = []
        for iid, (interactions, similarities) in tmp_iid_info.items():
            if len(interactions) == 0 and self.use_averages:
                pred = self._predict_default(iid if self.type == 'user' else uid)
            else:
                pred = self.aggregation_fn(interactions, similarities)

            if pred is not None:
                pred_list.append((pred, iid))

        return nlargest(n, pred_list)
