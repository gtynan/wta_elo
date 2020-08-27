from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from copy import deepcopy
import matplotlib.pyplot as plt

from .model import get_current_ability, get_surface_weights
from ..constants import PARAMS


def get_rankings(player_map: Dict[str, int],
                 player_abilities: np.ndarray, player_trend: np.ndarray, player_at_abilities: np.ndarray,
                 surfaces: List[str]) -> pd.DataFrame:
    """Creates current player rankings for specific surfaces

    Args:
        player_map (Dict[str): Mapping of players names to int position
        player_abilities (np.ndarray): Current player abilities
        player_trend (np.ndarray): Current player trends
        player_at_abilities (np.ndarray): All time max player abilities
        surfaces (List[str]): Surfaces abilities correspond to (assumes last column in abilities is Base)

    Returns:
        pd.DataFrame: Current rankings with column for each surface
    """
    rankings = pd.DataFrame(data=0, index=player_map.keys(), columns=surfaces)

    for surface in rankings.columns:
        s_weights, _ = get_surface_weights(np.isin(surfaces, surface), PARAMS['surface_weight'])
        rankings[surface] = get_current_ability(
            player_abilities, player_at_abilities, s_weights, PARAMS['all_time_weight'],
            player_trend, PARAMS['trend_weight'])

    return rankings.sort_values(by=surfaces[0], ascending=False)


def get_model_calibration(p: np.array, source: np.array) -> plt.figure:
    """Generate visual of maodel calibration

    Args:
        p (np.array): Predicted probabilities for the winner
        source (np.array): match source

    Returns:
        plt.figure: Model calibration visual
    """
    # 0.5 predcitions mess up bins
    source = source[p != 0.5]
    p = p[p != 0.5]

    probs = deepcopy(p)
    probs[probs < 0.5] = 1 - probs[probs < 0.5]

    fration_of_positives, mean_prob = calibration_curve(y_true=p >= 0.5, y_prob=probs, n_bins=10)

    f, ax = plt.subplots(figsize=(14, 7))

    ax.plot(mean_prob, fration_of_positives, "s-", label='Overall')

    fration_of_positives, mean_prob = calibration_curve(
        y_true=p[source == 'W'] >= 0.5, y_prob=probs[source == 'W'], n_bins=10)

    ax.plot(mean_prob, fration_of_positives, "s-", label='WTA')

    fration_of_positives, mean_prob = calibration_curve(
        y_true=p[source == 'I'] >= 0.5, y_prob=probs[source == 'I'], n_bins=10)

    ax.plot(mean_prob, fration_of_positives, "s-", label='ITF')

    ax.plot([0.5, 1], [0.5, 1], "k:", label="Perfectly calibrated")

    ax.legend(loc="lower right")
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')

    return f


def get_model_performance(p: np.array, source: np.array) -> str:
    """Formats model performance into string"""

    itf_predictions = p[source == 'I']
    wta_predictions = p[source == 'W']

    return f"""
Overall Total: {len(p)}
Overall accuracy: {len(p[p > 0.5]) / len(p)}
Overall brier score: {np.mean((1 - p)**2)}

WTA total: {len(wta_predictions)}
WTA accuracy: {len(wta_predictions[wta_predictions > 0.5]) / len(wta_predictions)}
WTA brier score: {np.mean((1 - wta_predictions)**2)}

ITF total: {len(itf_predictions)}
ITF accuracy: {len(itf_predictions[itf_predictions > 0.5]) / len(itf_predictions)}
ITF brier score: {np.mean((1 - itf_predictions)**2)}
    """
