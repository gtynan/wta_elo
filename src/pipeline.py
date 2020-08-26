import numpy as np
import pandas as pd
from datetime import datetime

from .constants import PIPELINE_DATA_FILE, YEAR_FROM, YEAR_TO, CLEAN_DATA_FILE_PATH, SURFACE_MAP, ROUND_ORDER, PARAMS, SOURCE_COL, J_SURFACE_COL, J_WINNER_COL, J_LOSER_COL, J_SCORE_COL, J_T_NAME, J_T_DATE, J_ROUND
from .data_ingestion import get_raw_games
from .data_cleaning import score_to_int, get_player_map, surface_to_one_hot, get_inferred_date
from .model import elo


def run():

    # scraping
    raw_data = get_raw_games(PIPELINE_DATA_FILE, YEAR_FROM, YEAR_TO)

    # cleaning
    raw_data['inferred_date'] = get_inferred_date(raw_data, J_T_NAME, J_T_DATE, J_ROUND, ROUND_ORDER)

    player_map, raw_data[['WID', 'LID']] = get_player_map(raw_data, J_WINNER_COL, J_LOSER_COL)

    raw_data = raw_data[raw_data[J_SURFACE_COL].notnull()]
    s_categories, s_data = surface_to_one_hot(raw_data[J_SURFACE_COL], SURFACE_MAP)
    raw_data[s_categories] = s_data

    raw_data[['WGames', 'WSets', 'LGames', 'LSets']] = score_to_int(raw_data[J_SCORE_COL])

    clean_data = raw_data[['inferred_date', 'WID', 'LID', *s_categories, 'WGames', 'WSets', 'LGames', 'LSets',
                           SOURCE_COL, 'tourney_name', 'winner_name', 'loser_name']].dropna(subset=['WGames', 'WSets', 'LGames', 'LSets'])

    clean_data = clean_data.sort_values(by='inferred_date').reset_index(drop=True)

    clean_data.to_csv(CLEAN_DATA_FILE_PATH.format(PIPELINE_DATA_FILE, YEAR_FROM, YEAR_TO))

    date_cutoff = datetime.strptime('20180101', '%Y%m%d')

    train_data = clean_data[clean_data['inferred_date'] < date_cutoff].copy(
        deep=True).sort_values(by='inferred_date').reset_index(drop=True)
    test_data = clean_data[clean_data['inferred_date'] >= date_cutoff].copy(
        deep=True).sort_values(by='inferred_date').reset_index(drop=True)

    player_abilities = np.full((len(player_map), len(s_categories) + 1), 1500.)
    games_played = np.zeros((len(player_map), len(s_categories) + 1))
    player_trend = np.zeros((len(player_map),))
    alltime_abilities = np.full((len(player_map), len(s_categories) + 1), 1500.)

    train_abilities, train_games, train_trend, train_alltime, train_p = elo(
        params=PARAMS, data=train_data, date_col='inferred_date', w_id_col='WID', w_games_col='WGames',
        w_sets_col='WSets', l_id_col='LID', l_games_col='LGames', l_sets_col='LSets', surface_cols=s_categories,
        itf_col=SOURCE_COL, player_abs=player_abilities, games_played=games_played, player_trend=player_trend,
        at_abilities=alltime_abilities)

    np.testing.assert_array_equal(player_abilities, 1500.)

    test_abilities, test_games, test_trend, test_alltime, test_predictions = elo(
        params=PARAMS, data=test_data, date_col='inferred_date', w_id_col='WID', w_games_col='WGames',
        w_sets_col='WSets', l_id_col='LID', l_games_col='LGames', l_sets_col='LSets', surface_cols=s_categories,
        itf_col=SOURCE_COL, player_abs=train_abilities, games_played=train_games, player_trend=train_trend,
        at_abilities=train_alltime)

    itf_predictions = test_predictions[test_data[SOURCE_COL] == 'I']
    wta_predictions = test_predictions[test_data[SOURCE_COL] == 'W']

    print(f"""Total: {len(test_predictions)}
        Overall accuracy: {len(test_predictions[test_predictions > 0.5]) / len(test_predictions)}
        Overall brier score: {np.mean((1 - test_predictions)**2)}

        WTA total: {len(wta_predictions)}
        WTA accuracy: {len(wta_predictions[wta_predictions > 0.5]) / len(wta_predictions)}
        WTA brier score: {np.mean((1 - wta_predictions)**2)}
        WTA loglik: {np.log(wta_predictions).sum()}
        
        ITF total: {len(itf_predictions)}
        ITF accuracy: {len(itf_predictions[itf_predictions > 0.5]) / len(itf_predictions)}
        ITF brier score: {np.mean((1 - itf_predictions)**2)}

        Train loglik: {(np.log(train_p)*(train_data[SOURCE_COL] == 'W')).sum()}""")

    test_data['p'] = test_predictions
    inv_map = {v: k for k, v in player_map.items()}

    test_data['winner_name'] = test_data['WID'].map(inv_map)
    test_data['loser_name'] = test_data['LID'].map(inv_map)

    test_data.to_csv('data/03_output/test_predictions.csv')
