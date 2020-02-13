# TODO List:
- Refactors tests for the bultin dbs submodule
- Tests for the evaluation metrics
- Tests for the recommenders self.n_users and self.n_items
- leave-k-out with split by timestamp option
- Support jester dataset (and datasets where min_rating != 0)
- How to save Recommenders? they depend on ratings dataset... is it possible to decouple? if not, what is the best way?
- Add NAN check for when bad delimiters are provided on interaction dataset imports
- Add feature to compute validation loss on the provided validation dataset during training
- Make evaluation processes sample test users instead of running each sequentially