## Macro-regime Analysis


### Parametri
# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

hp_to_analyze = 7


# --- LISTA TICKER ---
US_Equity = ["AA","AAL","AAPL","ABBV","ABT","ACN","ADBE","ADSK","ALGN","AMD","AMGN","AMZN","B","BA",
    "BABA","BAC","BBVA","BBY","BIDU","BLK","BMY","BNTX","BRK.B","C","CAT","CCL","CGC","CL",
    "CMCSA","COIN","COP","COST","CRM","CSCO","CTAS","CVS","CVX","DAL","DBX","DGX","DHI",
    "DIS","DLR","DOCU","DOV","DPZ","DRI","DTE","DVA","DVN","EBAY","ECL","ED","EFX","EIX",
    "EMN","EMR","EQR","ESS","ETR","ETSY","EXPD","F","FBIN","FCX","FE","FFIV","EXR","FIS",
    "FITB","FLS","FMC","FSLR","GE","GM","GOOGL","GS","HAS","HD","HON","HOOD","IBM","IDXX",
    "INTC","JNJ","JPM","KHC","KO","LLY","LMT","LUV","LYFT","MA","MCD","MCO","MDT","MELI",
    "META","MGM","MMM","MO","MS","MSFT","MU","NEE","NFLX","NKE","NVDA","PENN","PFE","PG",
    "PINS","PM","PYPL","QCOM","RCL","REGN","RTX","SBUX","SEDG","SHOP","SLB","SNAP","SPOT",
    "SU","SYK","T","TFX","TMUS","TSLA","TTWO","UAL","UBER","V","VZ","WFC","WMT","XOM",
    "XYZ","ZM"]

Indici_US_ETF = ["SPY","QQQ","IWM","DIA"]

Altri_ETF = ["AUMI","EEM","EWJ","EWW","FXI","GLD","IEMG","IWM","IYR","JNUG","MJ",
    "QQQ","SPY","SQQQ","TLT","XLB","XLE","XLK","XLU","XLY","XME","XOP"]

#tickers = US_Equity
#tickers = US_Equity + Indici_US_ETF + Altri_ETF
#tickers = ["ZM","SPY","AAPL","MSFT","NVDA","AMZN"]
tickers = Indici_US_ETF

# ---- x market filter ----
symbol_spy = qb.add_equity("SPY").symbol
history_spy = qb.history(symbol_spy, start_date, end_date, timeframe_security)

df_spy = history_spy.loc[symbol_spy].copy()

# Calcolo SMA200 e Above_SMA200 su SPY
df_spy["SMA200"] = df_spy["close"].rolling(200).mean()
df_spy["Above_SMA200"] = df_spy["close"] > df_spy["SMA200"]

# Assicurati che df_spy abbia una colonna 'date' per il merge
df_spy.reset_index(inplace=True)  # così la colonna 'date' sarà esplicita


---------
## Test Long
---------

def Oops_di_Williams_Long(df):

    patterns = [None, None]    

    for i in range(2, len(df)):

        o, h, l, c = map(float, df.iloc[i][["open","high","low","close"]])
        o_minus1, h_minus1, l_minus1, c_minus1 = map(float, df.iloc[i-1][["open","high","low","close"]])

        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        body_minus1 = abs(c_minus1 - o_minus1)
        total_range_minus1 = h_minus1 - l_minus1
        upper_shadow_minus1 = h_minus1- max(o_minus1, c_minus1)
        lower_shadow_minus1 = min(o_minus1, c_minus1) - l_minus1


# --- PATTERN ---
        # LONG
        if (
            o < l_minus1 * (1 - toll_gap) and
            h >= l_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono il low precedente
        ):
            patterns.append("Oops di williams Long")
        
        else:
            patterns.append(None)

    df['Pattern'] = patterns
    return df


# --- MAPPA LONG/SHORT ---
PATTERN_DIRECTION = {

    # long
    "Oops di williams Long": "LONG",
}


def collect_trades(df, hp_to_analyze):
    trades = []

    for i in range(1, len(df) - hp_to_analyze - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue
        
        side = PATTERN_DIRECTION[pattern]

        entry_price = df["low"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        highs = df["high"].iloc[i:i + hp_to_analyze -1] # -1 because I count the entry date as day 1
        lows  = df["low"].iloc[i:i + hp_to_analyze -1]
        closes = df["close"].iloc[i:i + hp_to_analyze -1]

        exit_price = closes.iloc[-1]

        # Return
        ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        positive_ret_pct = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        # >>> CONTEXT <<<
        sma200_slope = df["SMA200_slope"].iloc[i]
        dist_sma200 = df["Dist_SMA200"].iloc[i]
        atr_pct = df["ATR_pct"].iloc[i]
        sma5_slope = df["sma5_slope"].iloc[i]


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret_pct*100 if not np.isnan(positive_ret_pct) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": hp_to_analyze,
            "trade_date": entry_date,
            
            # Contesto
            "SMA200_slope": sma200_slope,
            "Dist_SMA200": dist_sma200,
            "ATR_pct": atr_pct,
            "sma5_slope": sma5_slope,
        })

    return trades


def compute_metrics_from_df(df):
    if len(df) == 0:
        return None

    returns = df["return_pct"].dropna()
    positive_returns = df["positive_return_pct"].dropna()
 
    metrics = {
        "n_trades": len(df),
        
        "Expectancy": returns.mean(),

        # ritorni
        "Median_ret_pct": returns.median(),
        #"p25_ret": returns.quantile(0.25),
        #"p75_ret": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "Median_pos_ret_pct": positive_returns.median(),
        #"p25_pos_ret": positive_returns.quantile(0.25),
        #"p75_pos_ret": positive_returns.quantile(0.75),
        #"Skew_pos_ret": positive_returns.skew(),

        # rischio
        "MAE_mean_pct": df["mae_pct"].mean(),
        "MAE_p75_pct": df["mae_pct"].quantile(0.75),
        #"MAE_p90": df["mae_pct"].quantile(0.90),

        "MFE_mean_pct": df["mfe_pct"].mean(),
        "MFE_p75_pct": df["mfe_pct"].quantile(0.75),
        #"MFE_p90": df["mfe_pct"].quantile(0.90),

        # edge dynamics
        "TMFE_mean": df["tmfe"].mean(),
    }

    return metrics


# --- MAIN LOOP ---
all_trades= []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    # --- CONTEXT INDICATORS ---
    df["SMA200"] = df["close"].rolling(200).mean()
    df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(1)

    #Distance from mean
    df["Dist_SMA200"] = (df["close"] - df["SMA200"]) / df["SMA200"]

    # Volatility
    df["TR"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["ATR14"] = df["TR"].rolling(14).mean()
    df["ATR_pct"] = df["ATR14"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])  # Percentile rolling su 50 giorni

    # Local Trend
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma5_slope"] = df["sma5"] - df["sma5"].shift(5)

    #df["adx14"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # Market Regime
    #spy["sma200"] = spy["close"].rolling(200).mean()
    #spy["above_sma200"] = spy["close"] > spy["sma200"]

    #spy["ret"] = spy["close"].pct_change()
    #spy["vol20"] = spy["ret"].rolling(20).std()


    trades = collect_trades(df, hp_to_analyze)
    all_trades.extend(trades)

trades_df = pd.DataFrame(all_trades)


# --- Parte I macro ---
df_trend = trades_df[trades_df["SMA200_slope"] > 0]
df_notrend = trades_df[trades_df["SMA200_slope"] <= 0]

df_above = trades_df[trades_df["Dist_SMA200"] > 0]    # partiamo dal trades_df così da non mischiare i regimi
df_below = trades_df[trades_df["Dist_SMA200"] <= 0]   # partiamo dal trades_df così da non mischiare i regimi

df_lowvol  = trades_df[trades_df["ATR_pct"] < 0.5]
df_highvol = trades_df[trades_df["ATR_pct"] >= 0.5]

df_trend_up = trades_df[trades_df["sma5_slope"] > 0]
df_trend_down = trades_df[trades_df["sma5_slope"] < 0]

# Faccio il merge con df_spy per associare Above_SMA200 al trade
trades_df = trades_df.merge(
    df_spy[['time', 'Above_SMA200']],
    left_on='trade_date',
    right_on='time',
    how='left'
)

df_market_up = trades_df[trades_df['Above_SMA200'] == True]
df_market_down = trades_df[trades_df['Above_SMA200'] == False]


# --- Parte I micro ----
df_trend_up = trades_df[trades_df["sma5_slope"] > 0]
df_trend_down = trades_df[trades_df["sma5_slope"] < 0]



# --- Parte II macro ---
metrics_trend = compute_metrics_from_df(df_trend)
metrics_notrend = compute_metrics_from_df(df_notrend)

metric_above = compute_metrics_from_df(df_above)
metric_below = compute_metrics_from_df(df_below)

metrics_lowvol  = compute_metrics_from_df(df_lowvol)
metrics_highvol = compute_metrics_from_df(df_highvol)

metrics_market_up = compute_metrics_from_df(df_market_up)
metrics_market_down = compute_metrics_from_df(df_market_down)

# --- Parte II micro ---
metrics_trend_up = compute_metrics_from_df(df_trend_up)
metrics_trend_down = compute_metrics_from_df(df_trend_down)



print("ok")


### Risultati Trend Vs NoTrend con SMA200 - LONG
summary_context = pd.DataFrame([
    {"Regime": "Trend",**metrics_trend},
    {"Regime": "NoTrend",**metrics_notrend}
])

display(summary_context.round(2))


### Risultati Above Vs Below da SMA200 - LONG
summary_context = pd.DataFrame([
    {"Regime": "Above",**metric_above},
    {"Regime": "Below",**metric_below}
])

display(summary_context.round(2))


### Risultati high volatility Vs low volatility - LONG
summary_context = pd.DataFrame([
    {"Regime": "Low Volatility",**metrics_lowvol},
    {"Regime": "High Volatility",**metrics_highvol}
])

display(summary_context.round(2))


### Risultati Market up vs market down - LONG
summary_context = pd.DataFrame([
    {"Regime": "Market Up",**metrics_market_up},
    {"Regime": "Market Down",**metrics_market_down}
])

display(summary_context.round(2))



### Risultati local trend up Vs local trend down
summary_context = pd.DataFrame([
    {"Regime": "Local Trend Up ",**metrics_trend_up},
    {"Regime": "LocalTrendDown ",**metrics_trend_down}
])

display(summary_context.round(2))


-----------
## Test Short
-----------
def Oops_di_Williams_Short(df):

    patterns = [None, None]

    for i in range(2, len(df)):

        o, h, l, c = map(float, df.iloc[i][["open","high","low","close"]])
        o_minus1, h_minus1, l_minus1, c_minus1 = map(float, df.iloc[i-1][["open","high","low","close"]])

        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        body_minus1 = abs(c_minus1 - o_minus1)
        total_range_minus1 = h_minus1 - l_minus1
        upper_shadow_minus1 = h_minus1- max(o_minus1, c_minus1)
        lower_shadow_minus1 = min(o_minus1, c_minus1) - l_minus1


# --- PATTERN ---
        if (
            o > h_minus1 * (1 + toll_gap) and
            l <= h_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono l' high precedente
        ):
            patterns.append("Oops di williams Short")
        else:
            patterns.append(None)

    df['Pattern'] = patterns
    return df


# --- MAPPA LONG/SHORT ---
PATTERN_DIRECTION = {
    # short
    "Oops di williams Short": "SHORT",
}


def collect_trades(df, hp_to_analyze):
    trades = []

    for i in range(1, len(df) - hp_to_analyze - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue
            
        side = PATTERN_DIRECTION[pattern]

        entry_price = df["high"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        highs = df["high"].iloc[i:i + hp_to_analyze-1]
        lows  = df["low"].iloc[i:i + hp_to_analyze-1]
        closes = df["close"].iloc[i:i + hp_to_analyze-1]

        exit_price = closes.iloc[-1]

        # Return
        ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        # >>> CONTEXT <<<
        sma200_slope = df["SMA200_slope"].iloc[i]
        dist_sma200 = df["Dist_SMA200"].iloc[i]
        atr_pct = df["ATR_pct"].iloc[i]
        sma5_slope = df["sma5_slope"].iloc[i]


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": hp_to_analyze,
            "trade_date": entry_date,
            
            # Contesto
            "SMA200_slope": sma200_slope,
            "Dist_SMA200": dist_sma200,
            "ATR_pct": atr_pct,
            "sma5_slope": sma5_slope,
        })

    return trades


def compute_metrics_from_df(df):
    if len(df) == 0:
        return None

    returns = df["return_pct"].dropna()
    positive_returns = df["positive_return_pct"].dropna()
 
    metrics = {
        "n_trades": len(df),
        
        "Expectancy": returns.mean(),

        # ritorni
        "Median_ret_pct": returns.median(),
        #"p25_ret": returns.quantile(0.25),
        #"p75_ret": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "Median_pos_ret_pct": positive_returns.median(),
        #"p25_pos_ret": positive_returns.quantile(0.25),
        #"p75_pos_ret": positive_returns.quantile(0.75),
        #"Skew_pos_ret": positive_returns.skew(),

        # rischio
        "MAE_mean_pct": df["mae_pct"].mean(),
        "MAE_p75_pct": df["mae_pct"].quantile(0.75),
        #"MAE_p90": df["mae_pct"].quantile(0.90),

        "MFE_mean_pct": df["mfe_pct"].mean(),
        "MFE_p75_pct": df["mfe_pct"].quantile(0.75),
        #"MFE_p90": df["mfe_pct"].quantile(0.90),

        # edge dynamics
        "TMFE_mean": df["tmfe"].mean(),
    }

    return metrics


# --- MAIN LOOP ---
all_trades= []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Short(df)

    # --- CONTEXT INDICATORS ---
    df["SMA200"] = df["close"].rolling(200).mean()
    df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(1)

    #Distance from mean
    df["Dist_SMA200"] = (df["close"] - df["SMA200"]) / df["SMA200"]

    # Volatility
    df["TR"] = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["ATR14"] = df["TR"].rolling(14).mean()
    df["ATR_pct"] = df["ATR14"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])  # Percentile rolling su 50 giorni

    # Local Trend
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma5_slope"] = df["sma5"] - df["sma5"].shift(5)

    #df["adx14"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # Market Regime
    #spy["sma200"] = spy["close"].rolling(200).mean()
    #spy["above_sma200"] = spy["close"] > spy["sma200"]

    #spy["ret"] = spy["close"].pct_change()
    #spy["vol20"] = spy["ret"].rolling(20).std()


    trades = collect_trades(df, hp_to_analyze)
    all_trades.extend(trades)

trades_df = pd.DataFrame(all_trades)


# --- Parte I macro ---
df_trend = trades_df[trades_df["SMA200_slope"] > 0]
df_notrend = trades_df[trades_df["SMA200_slope"] <= 0]

df_above = trades_df[trades_df["Dist_SMA200"] > 0]    # partiamo dal trades_df così da non mischiare i regimi
df_below = trades_df[trades_df["Dist_SMA200"] <= 0]   # partiamo dal trades_df così da non mischiare i regimi

df_lowvol  = trades_df[trades_df["ATR_pct"] < 0.5]
df_highvol = trades_df[trades_df["ATR_pct"] >= 0.5]

df_trend_up = trades_df[trades_df["sma5_slope"] > 0]
df_trend_down = trades_df[trades_df["sma5_slope"] < 0]

# Faccio il merge con df_spy per associare Above_SMA200 al trade
trades_df = trades_df.merge(
    df_spy[['time', 'Above_SMA200']],
    left_on='trade_date',
    right_on='time',
    how='left'
)

df_market_up = trades_df[trades_df['Above_SMA200'] == True]
df_market_down = trades_df[trades_df['Above_SMA200'] == False]


# --- Parte I micro ----
df_trend_up = trades_df[trades_df["sma5_slope"] > 0]
df_trend_down = trades_df[trades_df["sma5_slope"] < 0]



# --- Parte II macro ---
metrics_trend = compute_metrics_from_df(df_trend)
metrics_notrend = compute_metrics_from_df(df_notrend)

metric_above = compute_metrics_from_df(df_above)
metric_below = compute_metrics_from_df(df_below)

metrics_lowvol  = compute_metrics_from_df(df_lowvol)
metrics_highvol = compute_metrics_from_df(df_highvol)

metrics_market_up = compute_metrics_from_df(df_market_up)
metrics_market_down = compute_metrics_from_df(df_market_down)

# --- Parte II micro ---
metrics_trend_up = compute_metrics_from_df(df_trend_up)
metrics_trend_down = compute_metrics_from_df(df_trend_down)



print("ok")


### Risultati Trend Vs NoTrend con SMA200 - SHORT
summary_context = pd.DataFrame([
    {"Regime": "Trend",**metrics_trend},
    {"Regime": "NoTrend",**metrics_notrend}
])

display(summary_context.round(2))



### Risultati Above Vs Below da SMA200 - SHORT
summary_context = pd.DataFrame([
    {"Regime": "Above",**metric_above},
    {"Regime": "Below",**metric_below}
])

display(summary_context.round(2))



### Risultati high volatility Vs low volatility - SHORT
summary_context = pd.DataFrame([
    {"Regime": "Low Volatility",**metrics_lowvol},
    {"Regime": "High Volatility",**metrics_highvol}
])

display(summary_context.round(2))



### Risultati Market up vs market down - SHORT
summary_context = pd.DataFrame([
    {"Regime": "Market Up",**metrics_market_up},
    {"Regime": "Market Down",**metrics_market_down}
])

display(summary_context.round(2))



