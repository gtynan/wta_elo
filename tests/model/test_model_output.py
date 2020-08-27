import numpy as np

from src.model.model_output import get_rankings
from src.model.model import get_current_ability
from src.constants import PARAMS


def test_get_rankings():
    player_map = {"Tom": 0, "Dick": 1, "Harry": 2}
    player_abs = np.array([
        [100, 120],
        [200, 120],
        [100, 100]
    ])
    player_trend = np.array([0, 1, -1])
    player_at = np.array([
        [300, 300],
        [400, 400],
        [100, 100]
    ])
    surfaces = ['Hard']

    e_ranking = get_current_ability(
        player_abs, player_at, [PARAMS['surface_weight'],
                                1 - PARAMS['surface_weight']],
        PARAMS['all_time_weight'],
        player_trend, PARAMS['trend_weight'])

    rankings = get_rankings(player_map, player_abs, player_trend, player_at, surfaces)

    assert rankings.columns == surfaces
    np.testing.assert_array_equal(rankings.index, ["Dick", "Tom", "Harry"])
    assert np.isin(e_ranking, rankings.values).all()
