## Consistency Evaluation
The scripts present on this folder were used to compute the results of the consistency evaluation of this work.

### Baseline Recommenders
Paper used as reference for the comparison: is M. Barros, A. Moitinho and F. M. Couto, "Using Research Literature to Generate Datasets of Implicit Feedback for Recommending Scientific Items," in IEEE Access, vol. 7, pp. 176668-176680, 2019.

### Deep Recommenders
For the deep learning based recommenders, the DMF model is used, which is described in Xue, H.-J., Dai, X., Zhang, J., Huang, S., and Chen, J. (2017). Deep matrix factorization models for recommender systems. InIJCAI, pages 3203–3209.

There are two models present on this test, the DMF-CE and DMF-NCE, where the first one receives interaction values on their original scale, while the last variant receives binary interaction values (every interaction present on the data set is set to 1, and all the others are assumed to be 0). 