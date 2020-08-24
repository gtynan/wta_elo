from typing import Tuple, Dict, List
import pandas as pd
import numpy as np


def get_player_map(data: pd.DataFrame, w_col: str, l_col: str) -> Tuple[Dict[str, int], np.ndarray]:
    """Generates a dictionary containing each players name as key and a unique int as value, also returns winner and loser columns mapped to new int id

    Args:
        data (pd.DataFrame): data
        w_col (str): column containing winners name
        l_col (str): column containing losers name

    Returns:
        Tuple[[Dict[str, int], np.ndarray]: unique identifier for each player, array of winner and loser ids for each game
    """
    # generate unique id for each player
    player_map = {player: i for i, player in enumerate(
        np.unique(data[[w_col, l_col]].values.ravel())
    )}
    # map player to id
    w_ids = data[w_col].map(player_map)
    l_ids = data[l_col].map(player_map)

    return player_map, np.column_stack((w_ids, l_ids))


def score_to_int(score: pd.Series) -> np.ndarray:
    """Converts series of string scores to int values representing games and sets won by winner and loser.

    Args:
        score (pd.Series): series of scores

    Returns:
        np.ndarray: matrix containing [WGames, WSets, LGames, LSets] each row is a game 
    """
    # regex removes brackets and their contents (not interested in tiebreaks)
    score = score.str.replace(r'\([^)]*\)', '').str.strip()

    # each set denoted by space ie `6-0 6-1`
    # split and apply creates dataframe with column for each set
    set_df = score.str.split(' ').apply(pd.Series)

    # empty array to be filled with game scores
    game_scores = np.zeros((len(score), 6))
    # 0, 2, 4 winner cols and 1, 3, 5 loser cols
    w_set_cols, l_set_cols = range(0, 4, 2), range(1, 5, 2)

    # looping over each column (set score ie 6-0)
    for i, (_, set_series) in enumerate(set_df.iteritems()):
        i *= 2
        # df of w_game and l_games in set: 6-0 -> 6, 0
        set_games = set_series.str.split('-').apply(pd.Series)
        # errors coerce means if not numeric give null
        game_scores[:, i:i+2] = set_games.apply(pd.to_numeric, errors='coerce')

    # winning player must win at least 12 games otherwise score incorrect
    # we do not >= to catch NaNs
    game_scores[~(game_scores[:, w_set_cols].sum(axis=1) >= 12)] = np.NaN

    # counting number of sets won by winner
    w_sets = np.sum(game_scores[:, w_set_cols] > game_scores[:, l_set_cols], axis=1).astype(float)
    # winner must win at least 2 sets
    w_sets[w_sets < 2] = np.NaN

    return np.array([
        game_scores[:, w_set_cols].sum(axis=1),
        w_sets,
        game_scores[:, l_set_cols].sum(axis=1),
        # 2 - nan will return nan
        2 - w_sets]).T


def surface_to_one_hot(surfaces: pd.Series, surface_map: Dict[str, str]) -> Tuple[np.array]:
    """Convert surface to one hot encoding and limit categories to surface map values

    Args:
        surfaces (pd.Series): Series containing all playing surfaces
        surface_map (Dict[str, str]): Map indicating what to rename certain surfaces to

    Returns:
        Tuple[np.array]: surface categories, One hot surface encoding
    """
    surfaces = surfaces.map(surface_map)
    surface_categories = np.unique(list(surface_map.values()))
    return surface_categories, pd.get_dummies(pd.Categorical(surfaces, categories=surface_categories)).values


# series to preserve index (despite sorting within function)
def get_inferred_date(
        data: pd.DataFrame, t_name_col: str, t_date_col: str, round_col: str, round_order: List[str]
) -> pd.Series:

    # copying to ensure don't alter original
    data = data[[t_name_col, t_date_col, round_col]].copy(deep=True)
    # round as ordered categorical
    data['r_order'] = pd.Categorical(data[round_col], categories=round_order, ordered=True)
    # convert to date
    data[t_date_col] = pd.to_datetime(data[t_date_col], format='%Y%m%d')

    # low to high date then round
    data = data.sort_values(by=[t_date_col, 'r_order'])

    # this is where it gets complex to ensure speed

    # group by date and name to ensure same tournament,
    # pandas unique orders first come first served hence sort earlier means early round first later rounds after
    # apply series splits list in dataframe, group by is now index and columns are 0-> max number of rounds,
    # each tournament only has values as far as their max hence column value is round order
    # row for each unique tournament
    round_order_df = data.groupby([t_date_col, t_name_col])['r_order'].unique().apply(pd.Series)

    # map each row to their round order
    # by setting the group by columns as our original data index we can get row data just like a dictionary
    data_rounds = round_order_df.loc[data.set_index([t_date_col, t_name_col]).index].values

    # round val contains column data round was in, columns are numeric 0 -> and reflect position of round
    # relative to all rounds in said tournament
    _, round_val = np.where(data[round_col].values.reshape(-1, 1) == data_rounds)

    return data.loc[:, t_date_col] + pd.to_timedelta(round_val, unit='day')
