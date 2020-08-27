from typing import Dict, List
import numpy as np
import pandas as pd

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
