import pandas as pd

from .constants import RAW_DATA_FILE_PATH, WTA_URL, ITF_URL, SOURCE_COL


def get_raw_games(
        filename: str, year_from: int, year_to: int, save: bool = True) -> pd.DataFrame:
    """Gets all WTA and ITF games, given a filename will initially try and load if dataset exists otherwise will download and save

    Args:
        filename (str): name to save data as, (will attempt to load)
        year_from (int): starting year
        year_to (int): end year
        save (bool): save dataset after scraping (format can be seen in constants.py)

    Returns:
        pd.DataFrame: raw dataframe of games with additional column `source` indicating origin
    """
    file_path = RAW_DATA_FILE_PATH.format(filename, year_from, year_to)
    try:
        return pd.read_csv(file_path, index_col=0, encoding="ISO-8859-1", low_memory=False)
    except:
        data = None
        for year in range(year_from, year_to + 1):
            new_wta = pd.read_csv(WTA_URL.format(year), encoding="ISO-8859-1", low_memory=False)
            new_itf = pd.read_csv(ITF_URL.format(year), encoding="ISO-8859-1", low_memory=False)

            new_wta[SOURCE_COL] = 'W'
            new_itf[SOURCE_COL] = 'I'

            if isinstance(data, pd.DataFrame):
                data = data.append(new_wta, ignore_index=True)
            else:
                data = new_wta
            data = data.append(new_itf, ignore_index=True)

        if save:
            data.to_csv(file_path)
        return data
