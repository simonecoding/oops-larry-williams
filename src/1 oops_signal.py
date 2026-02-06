"""
Oops Pattern detection logic.

Implementation of the Oops pattern introduced by Larry Williams,
expressed in a fully systematic and testable form.
"""

import pandas as pd


def detect_oops_pattern(
    df: pd.DataFrame,
    gap_tolerance: float = 0.0
) -> pd.Series:
    """
    Detects Oops pattern occurrences (Long and Short).

    Long Oops:
    - Opening gap below previous low
    - Intraday price trades back to previous low

    Short Oops:
    - Opening gap above previous high
    - Intraday price trades back to previous high

    Parameters
    ----------
    df : pd.DataFrame
        OHLC price data with columns: open, high, low, close
    gap_tolerance : float
        Minimum gap threshold expressed as a percentage.

    Returns
    -------
    pd.Series
        Categorical series with values:
        - "LONG"
        - "SHORT"
        - None
    """

    patterns = [None, None]

    for i in range(2, len(df)):

        o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
        h_prev = df.iloc[i - 1]["high"]
        l_prev = df.iloc[i - 1]["low"]

        # Long Oops
        if (
            o < l_prev * (1 - gap_tolerance)
            and h >= l_prev
        ):
            patterns.append("LONG")

        # Short Oops
        elif (
            o > h_prev * (1 + gap_tolerance)
            and l <= h_prev
        ):
            patterns.append("SHORT")

        else:
            patterns.append(None)

    return pd.Series(patterns, index=df.index)
