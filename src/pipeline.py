import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .constants import PIPELINE_DATA_FILE, CLEAN_DATA_FILE_PATH, MODEL_OUTPUT_FOLDER, SURFACE_MAP, ROUND_ORDER, PARAMS, SOURCE_COL, J_SURFACE_COL, J_WINNER_COL, J_LOSER_COL, J_SCORE_COL, J_T_NAME, J_T_DATE, J_ROUND
from .data_ingestion.data_scraping import get_raw_games
from .data_ingestion.data_cleaning import score_to_int, get_player_map, surface_to_one_hot, get_inferred_date
from .model.model import elo
from .model.model_output import get_rankings, get_model_calibration, get_model_performance
from .logging_functions import timeit


@timeit
def run(year_from: int, year_to: int, test_size: int):
    """Runs pipeline
    """
    assert (year_to - year_from - test_size) > 0

    # scraping
    raw_data = get_raw_games(PIPELINE_DATA_FILE, year_from, year_to)

    # cleaning
    raw_data['inferred_date'] = get_inferred_date(raw_data, J_T_NAME, J_T_DATE, J_ROUND, ROUND_ORDER)

    player_map, raw_data[['WID', 'LID']] = get_player_map(raw_data, J_WINNER_COL, J_LOSER_COL)

    s_categories, s_data = surface_to_one_hot(raw_data[J_SURFACE_COL], SURFACE_MAP)
    raw_data[s_categories] = s_data

    score_cols = ['WGames', 'WSets', 'LGames', 'LSets']
    raw_data[score_cols] = score_to_int(raw_data[J_SCORE_COL])

    clean_data = raw_data[['inferred_date', 'WID', 'LID', *s_categories, *score_cols, SOURCE_COL,
                           'tourney_name', 'winner_name', 'loser_name']].dropna(subset=score_cols)

    clean_data = clean_data.sort_values(by='inferred_date').reset_index(drop=True)

    clean_data.to_csv(CLEAN_DATA_FILE_PATH.format(PIPELINE_DATA_FILE, year_from, year_to))

    logging.info('DATA CLEANING COMPLETE')

    # modelling
    logging.info('MODELING STARTED')

    date_cutoff = datetime.strptime(f'{year_to - test_size}1231', '%Y%m%d')

    train_data = clean_data[clean_data['inferred_date'] <= date_cutoff].copy(
        deep=True).sort_values(by='inferred_date').reset_index(drop=True)
    test_data = clean_data[clean_data['inferred_date'] > date_cutoff].copy(
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

    test_abilities, test_games, test_trend, test_alltime, test_predictions = elo(
        params=PARAMS, data=test_data, date_col='inferred_date', w_id_col='WID', w_games_col='WGames',
        w_sets_col='WSets', l_id_col='LID', l_games_col='LGames', l_sets_col='LSets', surface_cols=s_categories,
        itf_col=SOURCE_COL, player_abs=train_abilities, games_played=train_games, player_trend=train_trend,
        at_abilities=train_alltime)

    logging.info('MODELING COMPLETE')

    # output
    test_data['p'] = test_predictions
    test_data.to_csv(f'{MODEL_OUTPUT_FOLDER}test_predictions.csv')

    model_performance = get_model_performance(test_predictions, test_data[SOURCE_COL].values)
    with open(f'{MODEL_OUTPUT_FOLDER}test_model_performance.txt', 'w+') as txt:
        txt.write(model_performance)

    fitted_ranking = get_rankings(player_map, test_abilities, test_trend, test_alltime, s_categories)
    fitted_ranking.to_csv(f'{MODEL_OUTPUT_FOLDER}test_rankings.csv')

    calibration_fig = get_model_calibration(test_predictions, test_data[SOURCE_COL].values)
    calibration_fig.savefig(f'{MODEL_OUTPUT_FOLDER}test_model_calibration.png')
