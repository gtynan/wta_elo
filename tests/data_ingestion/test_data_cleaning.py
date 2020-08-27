import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_ingestion.data_cleaning import score_to_int, get_player_map, surface_to_one_hot, get_inferred_date
from src.constants import J_SURFACE_COL, SURFACE_MAP, J_T_DATE, J_T_NAME, J_ROUND, ROUND_ORDER


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
    scores = pd.Series(['7-6(7) 6-0', '6-0 RET', '1-0 1-0', np.NaN, '6-0 0-6 6-0'])
    result = score_to_int(scores)

    # only the first and last score should return valid values
    np.testing.assert_array_equal(result[0], [13, 2, 6, 0])
    np.testing.assert_array_equal(result[1:-1], np.NaN)
    np.testing.assert_array_equal(result[-1], [12, 2, 6, 1])


def test_surface_to_one_hot():

    surfaces = pd.Series(data=np.array(["Grass", "Hard", "Clay", "Carpet", "Hard", pd.NaT]))

    s_categories, s_one_hot = surface_to_one_hot(surfaces, SURFACE_MAP)
    t_data = pd.DataFrame(data=s_one_hot, columns=s_categories)

    assert t_data.loc[0, 'Grass'] == 1
    assert t_data.loc[1, 'Hard'] == 1
    assert t_data.loc[2, 'Clay'] == 1
    assert t_data.loc[3, 'Grass'] == 1
    assert t_data.loc[4, 'Hard'] == 1
    assert t_data.loc[5, 'Hard'] == 1
    assert all(t_data.sum(axis=1) == 1)


def test_get_inferred_date():
    start_date = datetime.strptime("20200101", '%Y%m%d')

    data = pd.DataFrame(data=np.array([
        [start_date, 'Test1', 'R16'],
        [start_date, 'Test2', 'R16'],
        [start_date, 'Test1', 'QF'],
        [start_date, 'Test1', 'SF'],
        [start_date, 'Test1', 'QF'],
    ]), index=[4, 2, 6, 1, 10], columns=[J_T_DATE, J_T_NAME, J_ROUND])

    # expected dates, func will reorder on dates but maintain index
    e_dates = pd.Series(data=np.array(
        [start_date,
         start_date,
         start_date + timedelta(days=1),
         start_date + timedelta(days=1),
         start_date + timedelta(days=2)],
        dtype='datetime64[ns]'), index=[4, 2, 6, 10, 1])

    dates = get_inferred_date(data, J_T_NAME, J_T_DATE, J_ROUND, ROUND_ORDER)
    pd.testing.assert_series_equal(dates, e_dates)
