from typing import Tuple, Dict
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
