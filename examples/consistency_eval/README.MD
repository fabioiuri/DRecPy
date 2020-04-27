## Consistency Evaluation
The scripts present on this folder were used to compute the results of the consistency evaluation section of the thesis.
However, the used data sets are not present here, since they would be too big to maintain on this repository.

There's one script for each data set, where we can find:
1. An initial data set loading instruction;
2. A split instructor that produces a train and test data set;
3. Model initialization and training;
4. Model evaluation.

Steps 3 and 4 repeat for each similarity function used during the training process (cosine, Jaccard, mean squared differences and Pearson correlation).

Paper used as reference for the comparison: is M. Barros, A. Moitinho and F. M. Couto, "Using Research Literature to Generate Datasets of Implicit Feedback for Recommending Scientific Items," in IEEE Access, vol. 7, pp. 176668-176680, 2019.
