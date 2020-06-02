from DRecPy.Recommender.Baseline.aggregation import mean
from DRecPy.Recommender.Baseline.aggregation import weighted_mean
import pytest


@pytest.fixture
def interactions():
    return [5, 2, 3, 1]


@pytest.fixture
def interactions_zeroes():
    return [0, 0, 0, 0]


@pytest.fixture
def similarities():
    return [1, 0.2, 0.1, 0.8]


@pytest.fixture
def similarities_zeroes():
    return [0, 0, 0, 0]


def test_mean_0(interactions):
    assert mean(interactions, None) == 2.75


def test_mean_1(interactions_zeroes):
    assert mean(interactions_zeroes, None) == 0


def test_mean_2():
    assert mean([], None) is None


def test_weighted_mean_0(interactions, similarities):
    assert round(weighted_mean(interactions, similarities), 4) == 3.0952


def test_weighted_mean_1(interactions_zeroes, similarities):
    assert weighted_mean(interactions_zeroes, similarities) == 0


def test_weighted_mean_2():
    assert weighted_mean([], []) is None


def test_weighted_mean_3(interactions, similarities_zeroes):
    assert weighted_mean(interactions, similarities_zeroes) is None
