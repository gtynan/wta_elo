from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import binom
from copy import deepcopy


def get_surface_weights(one_hot_surface: np.array, surface_weight: Union[float, np.array]) -> Tuple[np.array]:
    """Converts one hot surface encoding to weighted surface array with base weight appended as last column as well as a binary playing surface array

    Args:
        one_hot_surface (np.array): One hot surface encoding for each match (row)
        surface_weight (float or np.array): Weight to assign to surface, remainder to base

    Returns:
        Tuple[np.array]: surface weights, playing_surfaces
    """
    # only surface in play will be != 0
    s_weights = one_hot_surface*surface_weight
    try:
        # base takes all weight not assigned to surface, reshape so that 2d
        base_weight = 1 - s_weights.sum(axis=1).reshape(-1, 1)
        s_weight = np.append(s_weights, base_weight, axis=1)
    except np.AxisError:
        base_weight = 1 - s_weights.sum()
        s_weight = np.append(s_weights, base_weight)
    return s_weight, (s_weight > 0).astype(int)


def get_current_ability(
        c_ability: np.array, at_ability: np.array, s_weights: Union[float, np.array], at_weight: float, c_trend: float,
        trend_weight: float) -> float:
    """Converts players multiple current and all time abilities to single current ability based on weights provided

    Args:
        c_ability (np.array): Players current 4 abilities (3 surface, 1 Base), should be a row per person
        at_ability (np.array): Players all time best 4 abilities (3 surface, 1 Base), should be a row per person
        s_weights (Union[float, np.array]): Weights for each surface ability based on upcoming match
        at_weight (float): Weight assigned to all time ability
        c_trend (float): Current trend
        trend_weight (float): Weight (max % current ability can change by)

    Returns:
        float: Players current surface specific ability
    """
    assert c_ability.shape == at_ability.shape == s_weights.shape
    try:
        c_ab = np.sum(c_ability*s_weights, axis=1)*(1 - at_weight) + np.sum(at_ability*s_weights, axis=1)*at_weight
    except np.AxisError:
        c_ab = np.sum(c_ability*s_weights)*(1 - at_weight) + np.sum(at_ability*s_weights)*at_weight

    return c_ab + c_ab*c_trend*trend_weight


def get_probs(player_a: float, player_b: float) -> float:
    """Probability of player_a beating player_b

    Args:
        player_a (float): Ability of player a
        player_b (float): Ability of player b

    Returns:
        float: probability of player_a beating player_b
    """
    return 1 - 1/(1 + np.power(10, (player_a - player_b)/400))


def get_log_likelihood(probs: float, y: int) -> float:
    """Calculate log likelihood based on predictions

    Args:
        probs (float): predictions (0,1)
        y (int): outcomes, binary [0,1]

    Returns:
        float: log likelihood
    """
    return np.log(probs*y + (1 - y)*(1 - probs)).sum()


def get_performance_score(
        probs: float, g_won: int, g_lost: int, s_won: int, s_lost: int, p: float, s_boost: float) -> Tuple[np.array]:
    """Represents the winners performance in a game (losers = 1 - winners)

    Args:
        probs: (float): prior probability of the winner winning
        g_won (int): Games won by winner
        g_lost (int): Games lost by winner
        s_won (int): Sets won by winner
        s_lost (int): Sets lost by winner
        p (float): probability of winner winning a game
        s_boost (float): added to winners performance score if they win in staright sets

    Returns:
        Tuple[np.array]: Winners performance scores, Losers performance scores
    """
    w_performance = binom.cdf(g_won, (g_won + g_lost), p) + ((s_won - s_lost) == 2)*s_boost
    l_performance = 1 - w_performance

    return w_performance - probs, l_performance - (1 - probs)


def get_k_factor(games_played: int, K: float, offset: float, shape: float, ift_indicator: bool, itf_deduction: float) -> float:
    """K factor determines the magnitude a game can have on a players ability

    Args:
        games_played (int): Games played previously
        K (float): static number to be divided by
        offset (float): ensures K factor not too big to start
        shape (float): ensures K factor doesn't get too small over time
        ift_indicator (bool): True if game is played on itf circuit
        itf_deduction (float): % to deduct if game played on itf

    Returns:
        float: K factor
    """
    return K / np.power(games_played + offset, shape)*(1 - ift_indicator*itf_deduction)


def get_ability_change(
        k_factor: np.array, p_score: float, playing_surface: np.array) -> np.array:
    """Amount players ability to change based on most recent performance

    Args:
        k_factor (np.array): K factors for each surface (column), each row denotes a different player
        p_score (float): Players performance score in said game
        playing_surface (np.array): Surface match was played on and base

    Returns:
        np.array: Ability change for player on each surface (surfaces not played on will be 0)
    """
    assert k_factor.shape == playing_surface.shape

    return k_factor*p_score*playing_surface


def get_updated_trend(c_trend: float, performance: float, update_rate: float) -> float:
    """Update players current trend based on most recent performance

    Args:
        c_trend (float): Current trend
        performance (float): Most recent performance
        update_rate (float): EWMA rate

    Returns:
        float: updated trend
    """
    return c_trend*(1 - update_rate) + performance*update_rate


def elo_single_round(params: Dict[str, float],
                     data: pd.DataFrame,
                     w_id_col: str,
                     w_games_col: str,
                     w_sets_col: str,
                     l_id_col: str,
                     l_games_col: str,
                     l_sets_col: str,
                     surface_cols: List[str],
                     itf_col: str,
                     player_abs: np.array,
                     games_played: np.array,
                     player_trend: np.array,
                     at_abilities: np.array) -> Tuple[np.array]:
    """ELO runs over dates this function runs a single date and returns updated player specific data and predicted probs

    Args:
        params (Dict[str, float]): ELO model parameters (parameters.yml)
        data (pd.DataFrame): Match dataframe
        w_id_col (str): Column with winner ids
        w_games_col (str): Columns with games won by winner
        w_sets_col (str): Columns with sets won by winner
        l_id_col (str): Column with loser ids
        l_games_col (str): Columns with games won by loser
        l_sets_col (str): Columns with games won by loser
        surface_cols (List[str]): Columns of 1 hot encoding of surfaces
        itf_col (str): Column with bool indicator if game on itf circuit
        player_abs (np.array): current player abilities
        games_played (np.array): games played previously
        player_trend (np.array): current trend
        at_abilities (np.array): all time max abilities

    Returns:
        Tuple[np.array]: (UPDATED) player_abs, games_played, player_trend, at_abilities, probs
    """
    w_id, l_id = data[w_id_col].values, data[l_id_col].values

    s_weights, playing_surface = get_surface_weights(
        one_hot_surface=data[surface_cols].values, surface_weight=params['surface_weight'])

    # winner & loser current ability prior to game
    w_ability = get_current_ability(
        c_ability=player_abs[w_id],
        at_ability=at_abilities[w_id],
        s_weights=s_weights,
        at_weight=params['all_time_weight'],
        c_trend=player_trend[w_id],
        trend_weight=params['trend_weight'])

    l_ability = get_current_ability(
        c_ability=player_abs[l_id],
        at_ability=at_abilities[l_id],
        s_weights=s_weights,
        at_weight=params['all_time_weight'],
        c_trend=player_trend[l_id],
        trend_weight=params['trend_weight'])

    probs = get_probs(player_a=w_ability, player_b=l_ability)

    w_performance, l_performance = get_performance_score(
        probs=probs, g_won=data[w_games_col].values, g_lost=data[l_games_col].values, s_won=data[w_sets_col].values,
        s_lost=data[l_sets_col].values, p=params['p'],
        s_boost=params['straight_sets_boost'])

    itf_indicator = (data[itf_col] == 'I').values.reshape(-1, 1)

    w_k_factor = get_k_factor(
        games_played=games_played[w_id],
        K=params['K'],
        offset=params['offset'],
        shape=params['shape'],
        ift_indicator=itf_indicator,
        itf_deduction=params['itf_deduction'])

    l_k_factor = get_k_factor(
        games_played=games_played[l_id],
        K=params['K'],
        offset=params['offset'],
        shape=params['shape'],
        ift_indicator=itf_indicator,
        itf_deduction=params['itf_deduction'])

    player_abs[w_id] += get_ability_change(k_factor=w_k_factor,
                                           p_score=w_performance.reshape(-1, 1), playing_surface=playing_surface)
    player_abs[l_id] += get_ability_change(k_factor=l_k_factor,
                                           p_score=l_performance.reshape(-1, 1), playing_surface=playing_surface)

    # due to performance score although unlikely is possible losers ability increases and thus could be max observed
    at_abilities[w_id] = np.maximum(player_abs[w_id], at_abilities[w_id])
    at_abilities[l_id] = np.maximum(player_abs[l_id], at_abilities[l_id])

    player_trend[w_id] = get_updated_trend(c_trend=player_trend[w_id],
                                           performance=w_performance, update_rate=params['trend_rate'])
    player_trend[l_id] = get_updated_trend(c_trend=player_trend[l_id],
                                           performance=l_performance, update_rate=params['trend_rate'])

    games_played[w_id] += playing_surface
    games_played[l_id] += playing_surface

    return player_abs, games_played, player_trend, at_abilities, probs


def elo(params: Dict[str, float],
        data: pd.DataFrame,
        date_col: str,
        w_id_col: str,
        w_games_col: str,
        w_sets_col: str,
        l_id_col: str,
        l_games_col: str,
        l_sets_col: str,
        surface_cols: List[str],
        itf_col: str,
        player_abs: np.array,
        games_played: np.array,
        player_trend: np.array,
        at_abilities: np.array) -> Tuple[np.array]:
    """Main ELO model

    Args:
        params (Dict[str, float]): ELO model parameters (parameters.yml)
        data (pd.DataFrame): Match dataframe
        date_col (str): Column with date
         w_id_col (str): Column with winner ids
        w_games_col (str): Columns with games won by winner
        w_sets_col (str): Columns with sets won by winner
        l_id_col (str): Column with loser ids
        l_games_col (str): Columns with games won by loser
        l_sets_col (str): Columns with games won by loser
        surface_cols (List[str]): Columns of 1 hot encoding of surfaces
        itf_col (str): Column with bool indicator if game on itf circuit
        player_abs (np.array): current player abilities
        games_played (np.array): games played previously
        player_trend (np.array): current trend
        at_abilities (np.array): all time max abilities

    Returns:
        Tuple[np.array]: (UPDATED) player_abs, games_played, player_trend, at_abilities, probs (SORTED BY DATE)
    """

    # ensure not altering passed df and rely on sequential index to update arrays
    #data = data.copy(deep=True)

    player_abs = deepcopy(player_abs)
    games_played = deepcopy(games_played)
    player_trend = deepcopy(player_trend)
    at_abilities = deepcopy(at_abilities)

    overall_probs = np.empty((len(data),))

    for _, date_df in data.groupby(date_col):

        player_abs, games_played, player_trend, at_abilities, p = elo_single_round(
            params=params, data=date_df, w_id_col=w_id_col, w_games_col=w_games_col, w_sets_col=w_sets_col,
            l_id_col=l_id_col, l_games_col=l_games_col, l_sets_col=l_sets_col, surface_cols=surface_cols,
            itf_col=itf_col, player_abs=player_abs, games_played=games_played, player_trend=player_trend,
            at_abilities=at_abilities)

        overall_probs[date_df.index] = p

    return player_abs, games_played, player_trend, at_abilities, overall_probs
