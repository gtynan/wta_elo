import pytest
from os import path
from datetime import datetime

from src.data_ingestion import get_raw_games
from src.constants import PIPELINE_DATA_FILE, RAW_DATA_FILE_PATH


@pytest.mark.slow
def test_get_raw_games():
    # tourney date column in jeff sackmans dataset
    t_date_col = 'tourney_date'

    year_from, year_to = 2010, 2020
    # tests the loading, save=True for first time run
    # t_date in int form so still sorts correctly sans conversion to datetime
    data = get_raw_games(PIPELINE_DATA_FILE, year_from, year_to, save=True).sort_values(by=t_date_col)

    # this is the pipeline file so should already have been saved
    assert path.exists(RAW_DATA_FILE_PATH.format(PIPELINE_DATA_FILE, year_from, year_to))
    # convert tournament date to datetime and ensure correct dates were scraped
    assert datetime.strptime(str(data.iloc[0][t_date_col]), '%Y%m%d').year == year_from
    assert datetime.strptime(str(data.iloc[-1][t_date_col]), '%Y%m%d').year == year_to

    # tests the scraping
    data = get_raw_games('test', 2020, 2020, save=False).sort_values(by=t_date_col)
    assert datetime.strptime(str(data.iloc[0][t_date_col]), '%Y%m%d').year == 2020
    assert datetime.strptime(str(data.iloc[-1][t_date_col]), '%Y%m%d').year == 2020
