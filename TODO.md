# TODO List:
- Tests for the recommenders self.n_users and self.n_items
- How to save Recommenders? they depend on ratings dataset... is it possible to decouple? if not, what is the best way?
- Add NAN check for when bad delimiters are provided on interaction dataset imports
- Make evaluation processes sample test users instead of running each sequentially
- Add benchmark section to the readme
- Add item->iid and user->uid query optimizaztion to the db interaction dataset (already done on mem)
- Add better structure to the Splits submodule, so that its easier to understand what values are passed through during evaluation processes.
- Add better structure to the Processes submodule, so that its easier to understand what values are passed through during evaluation processes.
