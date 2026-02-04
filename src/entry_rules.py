def get_entry_price(df, index, side):
    """
    Entry logic based on previous session extremes.

    LONG  -> previous low
    SHORT -> previous high
    """

    if side == "LONG":
        return df["low"].iloc[index - 1]

    if side == "SHORT":
        return df["high"].iloc[index - 1]

    return None
