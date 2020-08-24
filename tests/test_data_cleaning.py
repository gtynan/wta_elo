import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import score_to_int, get_player_map


def test_get_player_map():
    players = np.array(['Tom', 'Dick', 'Harry'])
    w_col, l_col = 'WName', 'LName'

    data = pd.DataFrame(data=np.array([
        [players[0], players[1]],
        [players[2], players[1]],
    ]), columns=[w_col, l_col])

    player_map, games = get_player_map(data, w_col, l_col)

    assert len(player_map) == len(players)
    assert np.isin(players, list(player_map.keys())).all()
    assert np.max(list(player_map.values())) == 2
    assert np.min(list(player_map.values())) == 0

    # ensure ids correctly mapped
    np.testing.assert_array_equal(games[0], [player_map[players[0]], player_map[players[1]]])
    np.testing.assert_array_equal(games[1], [player_map[players[2]], player_map[players[1]]])


def test_score_to_int():
    scores = pd.Series(['7-6(7) 6-0', '6-0 RET', '1-0 1-0', np.NaN])
    result = score_to_int(scores)

    # only the first score should return valid values
    np.testing.assert_array_equal(result[0], [13, 2, 6, 0])
    np.testing.assert_array_equal(result[1:], np.NaN)
