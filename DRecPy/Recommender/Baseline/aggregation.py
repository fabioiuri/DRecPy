def mean(neighbours, interactions, similarities):
    """Computes the mean interaction of the user (if user_based == true) or the mean
    interaction of the item (item user_based == false). It simply sums the interaction
    values of the neighbours and divides by the total number of neighbours."""
    count, interaction_sum = 0, 0

    for neighbour, interaction, similarity in zip(neighbours, interactions, similarities):
        interaction_sum += interaction
        count += 1

    return interaction_sum / count if count > 0 else None


def weighted_mean(neighbours, interactions, similarities):
    """Computes the mean interaction of the user (if user_based == true) or the mean
   interaction of the item (item user_based == false). It computes the sum of the similarities
   multiplied by the interactions of each neighbour, and then divides this sum by the sum of
   the similarities of the neighbours."""
    sim_sum, interaction_sum = 0, 0

    for neighbour, interaction, similarity in zip(neighbours, interactions, similarities):
        interaction_sum += similarity * interaction
        sim_sum += similarity

    return interaction_sum / sim_sum if sim_sum > 0 else None
