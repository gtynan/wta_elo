import numpy as np
import pandas as pd

from .data_ingestion import get_raw_games
from .data_cleaning import score_to_int, get_player_map, surface_to_one_hot, get_inferred_date
from .constants import PIPELINE_DATA_FILE, CLEAN_DATA_FILE_PATH, SURFACE_MAP, ROUND_ORDER,  J_SURFACE_COL, J_WINNER_COL, J_LOSER_COL, J_SCORE_COL, J_T_NAME, J_T_DATE, J_ROUND


def run():
    try:
        clean_data = pd.read_csv(CLEAN_DATA_FILE_PATH.format(PIPELINE_DATA_FILE, 2010, 2020),
                                 parse_dates=['inferred_date'], low_memory=False)
    except:
        # scraping
        raw_data = get_raw_games(PIPELINE_DATA_FILE, 2010, 2020)

        # cleaning
        raw_data['inferred_date'] = get_inferred_date(raw_data, J_T_NAME, J_T_DATE, J_ROUND, ROUND_ORDER)

        player_map, raw_data[['WID', 'LID']] = get_player_map(raw_data, J_WINNER_COL, J_LOSER_COL)

        s_categories, s_data = surface_to_one_hot(raw_data[J_SURFACE_COL], SURFACE_MAP)
        raw_data[s_categories] = s_data

        raw_data[['WGames', 'WSets', 'LGames', 'LSets']] = score_to_int(raw_data[J_SCORE_COL])

        clean_data = raw_data[['inferred_date', 'WID', 'LID', 'WGames', 'WSets',
                               'LGames', 'LSets']].dropna(subset=['WGames', 'WSets', 'LGames', 'LSets'])

        clean_data.to_csv(CLEAN_DATA_FILE_PATH.format(PIPELINE_DATA_FILE, 2010, 2020))


run()
