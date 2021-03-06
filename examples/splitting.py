from DRecPy.Evaluation.Splits import leave_k_out
from DRecPy.Dataset import get_full_dataset
import time

dataset = get_full_dataset("ml-100k")
print('Full dataset', dataset)

# Dataset is split by leaving k user interactions out from the train set.
# If a given user does not have k interactions, all interactions stay on train set.
# Although, if a given user has < min_user_interactions, it will be removed
# from both sets.
start_t = time.time()
dataset_train, dataset_test = leave_k_out(dataset, k=10, min_user_interactions=20)
print(f'Splitting complete. Took: {time.time() - start_t}s')
print('Train dataset', dataset_train)
print('Test dataset', dataset_test)