import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import score_to_int


def test_score_to_int():
    scores = pd.Series(['7-6(7) 6-0', '6-0 RET', '1-0 1-0', np.NaN])

    # wraps 4 returned vectors as a matrix, each row is a returned vector
    result = np.array(score_to_int(scores))

    # only the first score should return valid values
    np.testing.assert_array_equal(result[:, 0], [13, 2, 6, 0])
    np.testing.assert_array_equal(result[:, 1:], np.NaN)
