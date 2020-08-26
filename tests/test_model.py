import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.constants import PARAMS
from src.model import (
    get_probs, get_k_factor, get_surface_weights, get_current_ability, get_performance_score, get_log_likelihood,
    get_ability_change, get_updated_trend, elo_single_round)


@pytest.mark.parametrize("one_hot_array, surface_weight, expected",
                         [(np.array([1, 0, 0]),
                           0.12,
                           (np.array([0.12, 0, 0, 0.88]),
                            np.array([1, 0, 0, 1]))
                           ),
                          (np.array([[1, 0, 0],
                                     [0, 1, 0]]),
                           0.12,
                           (np.array([[0.12, 0, 0, 0.88],
                                      [0, 0.12, 0, 0.88]]),
                            np.array([[1, 0, 0, 1],
                                      [0, 1, 0, 1]]))
                           )])
def test_get_surface_weights(one_hot_array, surface_weight, expected):
    np.testing.assert_array_equal(get_surface_weights(one_hot_array, surface_weight), expected)


@pytest.mark.parametrize("c_ability, at_ability, s_weights, at_weight, trend, trend_weight, expected",
                         [(np.array([2000, 2000, 2000, 2000]),
                           np.array([2200, 2000, 2000, 2200]),
                           np.array([0.12, 0, 0, .88]),
                           0.5,
                           .2,
                           .1,
                           2142)])
def test_get_current_ability(c_ability, at_ability, s_weights, at_weight, trend, trend_weight, expected):
    assert get_current_ability(c_ability, at_ability, s_weights, at_weight, trend,
                               trend_weight) == expected


@pytest.mark.parametrize("player_a, player_b, expected",
                         [(100, 100, .5),
                          (200, 100, 0.64),
                          (100, 200, 0.36)])
def test_get_probs(player_a, player_b, expected):
    assert get_probs(player_a, player_b) == pytest.approx(expected, 0.001)


@ pytest.mark.parametrize("probs, y, expected",
                          [(.8, 1, -0.223),
                           (np.array([0.9, 0.9]), np.array([1, 1]), -0.21),
                           (np.array([0.9, 0.9]), np.array([0, 1]), -2.41)])
def test_get_log_likelihood(probs, y, expected):
    assert get_log_likelihood(probs, y) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize("probs, g_won, g_lost, s_won, s_lost, p, s_boost, expected",
                         [(0.5, 12, 10, 2, 1, 0.5, .1, (0.238, -0.238)),
                          (0.6, -1, -1, -1, -1, 0.5, .1, (0.4, -0.4)),
                          (np.array([0.6, 0.7, 0.2]),
                           np.array([12, 12, -1]),
                           np.array([10, 2, -1]),
                           np.array([2, 2, -1]),
                           np.array([1, 0, -1]),
                           np.array([0.5, 0.55, 0.5]),
                           np.array([.1, .1, .1]),
                           (np.array([0.138, 0.397, .8]),
                            np.array([-0.138, -0.397, -.8])))])
def test_get_performance_score(probs, g_won, g_lost, s_won, s_lost, p, s_boost, expected):
    # expected generated from https://stattrek.com/online-calculator/binomial.aspx + boost if straight sets
    w_p, l_p = get_performance_score(probs, g_won, g_lost, s_won, s_lost, p, s_boost)

    np.testing.assert_array_almost_equal(w_p, expected[0], decimal=3)
    np.testing.assert_array_almost_equal(l_p, expected[1], decimal=3)


@ pytest.mark.parametrize("games_played, k, offset, shape, itf_indicator, itf_deduction, expected",
                          [(1, 250, 5, 0.4, True, 0.2, 97.67),
                           (10, 250, 5, 0.4, False, 0.2, 84.63)])
def test_get_k_factor(games_played, k, offset, shape, itf_indicator, itf_deduction, expected):
    assert get_k_factor(games_played, k, offset, shape, itf_indicator, itf_deduction) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize("k_factor, performance, playing_surface, expected",
                         [(np.array([100, 90, 80, 60]), np.array([.4]).reshape(-1, 1), np.array([1, 0, 0, 1]), np.array([40., 0., 0., 24.])),
                          (np.array([[100, 90, 80, 60], [80, 40, 20, 60]]),
                           np.array([.2, -.2]).reshape(-1, 1),
                           np.array([[1, 0, 0, 1], [1, 0, 0, 1]]),
                           np.array([[20., 0., 0., 12.], [-16., 0., 0., -12]]))
                          ])
def test_get_ability_change(k_factor, performance, playing_surface, expected):
    np.testing.assert_array_almost_equal(
        get_ability_change(k_factor, performance, playing_surface),
        expected.reshape(-1, 4),
        decimal=6)


@pytest.mark.parametrize("c_trend, performance, rate, expected",
                         [(0, 1, .2, .2), (.3, -.2, .5, 0.05)])
def test_get_updated_trend(c_trend, performance, rate, expected):
    assert get_updated_trend(c_trend, performance, rate) == pytest.approx(expected)
