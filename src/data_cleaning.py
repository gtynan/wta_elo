from typing import Tuple
import pandas as pd
import numpy as np


def score_to_int(score: pd.Series) -> Tuple[np.ndarray]:
    """Converts series of string scores to int values representing games and sets won by winner and loser

    Args:
        score (pd.Series): series of scores

    Returns:
        Tuple[np.array]: WGames, WSets, LGames, LSets
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
    game_scores[~(game_scores[:, w_set_cols].sum(axis=1) >= 12)] = np.NaN

    # counting number of sets won by winner
    w_sets = np.sum(game_scores[:, w_set_cols] > game_scores[:, l_set_cols], axis=1).astype(float)
    # winner must win at least 2 sets
    w_sets[w_sets < 2] = np.NaN

    return game_scores[:, w_set_cols].sum(axis=1), w_sets, game_scores[:, l_set_cols].sum(axis=1), 2 - w_sets
