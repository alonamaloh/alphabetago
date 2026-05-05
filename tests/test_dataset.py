import numpy as np

from alphabetago.dataset import featurize_games
from alphabetago.features import N_PLANES
from alphabetago.selfplay import play_random_game


def test_featurize_games_shapes():
    games = [play_random_game(size=5, seed=i) for i in range(3)]
    feats, policies, ownerships = featurize_games(games, n_workers=2)
    expected_n = sum(len(g.moves) for g in games)
    assert feats.shape == (expected_n, N_PLANES, 5, 5)
    assert policies.shape == (expected_n,)
    assert ownerships.shape == (expected_n, 5, 5)
    assert feats.dtype == np.int8
    assert policies.dtype == np.int32
    assert ownerships.dtype == np.int8


def test_featurize_values_in_range():
    games = [play_random_game(size=5, seed=42)]
    feats, policies, ownerships = featurize_games(games, n_workers=1)
    # Features are 0/1 indicators (cast to int8).
    assert ((feats == 0) | (feats == 1)).all()
    # Policy index in [0, n*n].
    assert (policies >= 0).all() and (policies <= 25).all()
    # Ownership in {-1, 0, +1}.
    assert ((ownerships >= -1) & (ownerships <= 1)).all()
