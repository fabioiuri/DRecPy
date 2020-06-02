from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import diags
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math


def cosine_sim(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the cosine
    similarities between each row to every other row of the provided matrix."""
    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    return cosine_similarity(matrix, dense_output=False)


def cosine_sim_cf(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the cosine
    similarities between each row to every other row of the provided matrix. The applied
    cosine formula is a variant that is sometimes considered on collaborative filtering (CF)
    environments (Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of
    recommender systems: A survey of the state-of-the-art and possible extensions. IEEE
    transactions on knowledge and data engineering, 17(6), 734-749. equation 13), by setting
    the denominator to only take into account the co-ratings of 2 comparing users/items."""
    if type(matrix) is not lil_matrix:
        matrix = lil_matrix(matrix)

    n = matrix.shape[0]
    rows, cols, data = [], [], []
    user_items = [sorted([(item, idx) for idx, item in enumerate(matrix.rows[i])]) for i in range(n)]

    for i in range(n):
        i_ratings, i_items = matrix.data[i], user_items[i]
        for j in range(i, n):
            j_ratings, j_items = matrix.data[j], user_items[j]
            sum_numerator, sum_denominator_i, sum_denominator_j = 0, 0, 0
            i_item_ctd, j_item_ctd = 0, 0
            while i_item_ctd < len(i_items) and j_item_ctd < len(j_items):
                if i_items[i_item_ctd][0] > j_items[j_item_ctd][0]:
                    j_item_ctd += 1
                elif i_items[i_item_ctd][0] < j_items[j_item_ctd][0]:
                    i_item_ctd += 1
                else:
                    i_idx = i_items[i_item_ctd][1]
                    j_idx = j_items[j_item_ctd][1]
                    sum_numerator += i_ratings[i_idx] * j_ratings[j_idx]
                    sum_denominator_i += i_ratings[i_idx] ** 2
                    sum_denominator_j += j_ratings[j_idx] ** 2
                    i_item_ctd += 1
                    j_item_ctd += 1

            if sum_numerator == 0: continue
            s = sum_numerator / (math.sqrt(sum_denominator_i) * math.sqrt(sum_denominator_j))
            rows.append(i), cols.append(j), data.append(s)
            if i != j: rows.append(j), cols.append(i), data.append(s)

    return csr_matrix((data, (rows, cols)))


def adjusted_cosine_sim(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the adjusted
    cosine similarities between each row to every other row of the provided matrix.
    Adjusted cosine sim. is the cosine sim. applied after subtracting the matrix column
    averages of each column values."""
    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    matrix = _subtract_row_mean(matrix)
    return cosine_similarity(matrix, dense_output=False)


def jaccard_sim(matrix):
    """Given a matrix NxM, returns a new matrix with size NxN containing all the jaccard
    similarities between each row to every other row of the provided matrix."""

    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    def matrix_jacc(matrix):
        matrix = matrix.astype(bool).astype(int)
        intersection = matrix.dot(matrix.T)
        row_sums = intersection.diagonal()
        row_sums = row_sums[:, None] + row_sums
        return csr_matrix(intersection / (row_sums - intersection))

    def iterative_jacc(matrix):
        matrix = lil_matrix(matrix)
        n = matrix.shape[0]
        rows, cols, data = [], [], []
        user_items = [sorted(matrix.rows[i]) for i in range(n)]

        for i in range(n):
            i_ratings, i_items = matrix.data[i], user_items[i]
            for j in range(i, n):
                j_ratings, j_items = matrix.data[j], user_items[j]
                common = 0
                i_item_ctd, j_item_ctd = 0, 0
                while i_item_ctd < len(i_items) and j_item_ctd < len(j_items):
                    if i_items[i_item_ctd] > j_items[j_item_ctd]:
                        j_item_ctd += 1
                    elif i_items[i_item_ctd] < j_items[j_item_ctd]:
                        i_item_ctd += 1
                    else:
                        common += 1
                        i_item_ctd += 1
                        j_item_ctd += 1

                if common == 0: continue
                s = common / (len(i_items) + len(j_items) - common)
                rows.append(i), cols.append(j), data.append(s)
                if i != j: rows.append(j), cols.append(i), data.append(s)

        return csr_matrix((data, (rows, cols)))

    try:
        return matrix_jacc(matrix)
    except MemoryError:
        return iterative_jacc(matrix)


def pearson_corr(matrix):
    """https://grouplens.org/blog/similarity-functions-for-user-user-collaborative-filtering/"""
    if type(matrix) is not lil_matrix:
        matrix = lil_matrix(matrix)

    n = matrix.shape[0]
    rows, cols, data = [], [], []
    user_items = [sorted([(item, idx) for idx, item in enumerate(matrix.rows[i])]) for i in range(n)]

    for i in range(n):
        i_ratings, i_items = matrix.data[i], user_items[i]
        for j in range(i, n):
            j_ratings, j_items = matrix.data[j], user_items[j]
            common_ratings = []  # list of tuples (i_rating, j_rating)
            avg_ratings_i, avg_ratings_j = 0, 0
            i_item_ctd, j_item_ctd = 0, 0
            while i_item_ctd < len(i_items) and j_item_ctd < len(j_items):
                if i_items[i_item_ctd][0] > j_items[j_item_ctd][0]:
                    j_item_ctd += 1
                elif i_items[i_item_ctd][0] < j_items[j_item_ctd][0]:
                    i_item_ctd += 1
                else:
                    i_idx = i_items[i_item_ctd][1]
                    j_idx = j_items[j_item_ctd][1]
                    common_ratings.append((i_ratings[i_idx], j_ratings[j_idx]))
                    avg_ratings_i += i_ratings[i_idx]
                    avg_ratings_j += j_ratings[j_idx]
                    i_item_ctd += 1
                    j_item_ctd += 1

            if len(common_ratings) == 0: continue

            avg_ratings_i /= len(common_ratings)
            avg_ratings_j /= len(common_ratings)
            common_ratings = [(i_r - avg_ratings_i, j_r - avg_ratings_j) for i_r, j_r in common_ratings]
            sum_numerator, sum_denominator_i, sum_denominator_j = 0, 0, 0
            for i_r, j_r in common_ratings:
                sum_numerator += i_r * j_r
                sum_denominator_i += i_r ** 2
                sum_denominator_j += j_r ** 2

            if sum_denominator_i == 0 or sum_denominator_j == 0: continue

            s = sum_numerator / (math.sqrt(sum_denominator_i) * math.sqrt(sum_denominator_j))
            rows.append(i), cols.append(j), data.append(s)
            if i != j: rows.append(j), cols.append(i), data.append(s)

    return csr_matrix((data, (rows, cols)))


def msd(matrix):
    if type(matrix) is not csr_matrix:
        matrix = csr_matrix(matrix)

    max_diff = matrix.max() - matrix.min()

    if type(matrix) is not lil_matrix:
        matrix = lil_matrix(matrix)

    n = matrix.shape[0]
    rows, cols, data = [], [], []
    user_items = [sorted([(item, idx) for idx, item in enumerate(matrix.rows[i])]) for i in range(n)]

    for i in range(n):
        i_ratings, i_items = matrix.data[i], user_items[i]
        for j in range(i, n):
            j_ratings, j_items = matrix.data[j], user_items[j]
            msd, common = 0, 0
            i_item_ctd, j_item_ctd = 0, 0
            while i_item_ctd < len(i_items) and j_item_ctd < len(j_items):
                if i_items[i_item_ctd][0] > j_items[j_item_ctd][0]:
                    j_item_ctd += 1
                elif i_items[i_item_ctd][0] < j_items[j_item_ctd][0]:
                    i_item_ctd += 1
                else:
                    i_idx = i_items[i_item_ctd][1]
                    j_idx = j_items[j_item_ctd][1]
                    msd += ((i_ratings[i_idx] - j_ratings[j_idx]) / max_diff) ** 2
                    common += 1
                    i_item_ctd += 1
                    j_item_ctd += 1

            if common == 0: continue
            s = 1 - msd / common
            rows.append(i), cols.append(j), data.append(s)
            if i != j: rows.append(j), cols.append(i), data.append(s)

    return csr_matrix((data, (rows, cols)))


def _subtract_row_mean(A):
    """For each row in the given matrix, subtracts the row mean from every value in the row.

    Args:
        A: csr_matrix.

    Returns:
        A row-modified csr_matrix.
    """
    assert type(A) is csr_matrix, "The given argument should be of type scipy.sparse.csr_matrix"

    sum_rows = np.array(A.sum(axis=1).squeeze())[0]
    size_rows = np.diff(A.indptr)
    avg_rows = np.divide(sum_rows, size_rows, where=size_rows != 0)
    avg_diag_matrix = diags(avg_rows, 0)
    ones_matrix = A.copy()
    ones_matrix.data = np.ones_like(A.data)

    return A - avg_diag_matrix * ones_matrix
