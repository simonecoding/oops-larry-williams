Filosofia di monetizzazione


-----------
-----------
## Price-Based Trailing
-----------
-----------


### Parametri

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

tol_dir = 0.2   # Tollereranza per candela direzionale = 20% 
min_body_ratio =  0.6 # Tollereranza per candela direzionale = body/ total range --> 60 % of body

toll_gap = 0.00   # esempio 0.002 = 0.2%

holding_periods = [3,4,5,6,7,10]

trailing_levels = [0.05, 0.075, 0.10, 0.20]


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


----------
### Test su Low
----------

def Oops_di_Williams_Long(df):

    patterns = [None, None]

    # BODY STRENGTH
    df["Body"] = (df["close"] - df["open"]).abs()
    df["Range"] = df["high"] - df["low"]
    df["Body_to_Range"] = df["Body"] / df["Range"].replace(0, np.nan)

    body_class = []

    for ratio in df["Body_to_Range"]:
        if pd.isna(ratio):
            body_class.append(None)
        elif ratio >= 0.7:
            body_class.append("long body")
        elif ratio >= 0.4:
            body_class.append("average body")
        else:
            body_class.append("small body")

    df["body_class"] = body_class

    # RANGE QUANTILI
    df["Total_Range"] = df["high"] - df["low"]

    range_class = []

    for i in range(len(df)):
        if i < 100:
            range_class.append(None)
        else:
            recent = df["Total_Range"].iloc[i-100:i]
            q33 = np.nanpercentile(recent, 33)
            q66 = np.nanpercentile(recent, 66)
            tr = df["Total_Range"].iloc[i]
            if tr < q33:
                range_class.append("smaller-sized")
            elif tr > q66:
                range_class.append("larger-sized")
            else:
                range_class.append("average")
    
    df["Range_Class"] = range_class
    

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

        body_class = df["body_class"].iloc[i]
        body_class_minus1 = df["body_class"].iloc[i-1]
    
        range_class = df["Range_Class"].iloc[i]
        range_class_minus1 = df["Range_Class"].iloc[i-1]


# --- PATTERN ---
        # LONG con pressione ribassista media e forte
        if (
            # 1° candle
            range_class_minus1 in ["larger-sized","average"] and
            body_class_minus1 in ["long body", "average body"] and
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
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


def collect_trades_trailing_low(df, holding_period, trailing_pct):
    trades = []

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        highs = df["high"].iloc[i:i+holding_period-1]
        lows  = df["low"].iloc[i:i+holding_period-1]
        closes  = df["close"].iloc[i:i+holding_period-1]

        max_price = entry_price
        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(len(highs)):
            current_high = highs.iloc[j]
            current_low  = lows.iloc[j]

            # aggiorna massimo favorevole
            if current_high > max_price:
                max_price = current_high

            trailing_level = max_price * (1 - trailing_pct)

            # condizione di uscita
            if current_low < trailing_level:
                exit_price = trailing_level # attenzione: prendo come prezzo di uscita il trailing level (sono intraday)!!
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # se non è mai scattato il trailing → exit time-based
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"

        # Return
        ret = (exit_price - entry_price) / entry_price

        # Positive return
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # %MFE captured-mean
        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "trailing_pct": trailing_pct,
            "exit_bar": exit_index,
            "exit_reason":exit_reason,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
    returns = df["return"].dropna()
    positive_returns = df["positive_return"].dropna()
    negative_retunrs = df.loc[df["return"] < 0, "return"]

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    # % MFE captured
    mfe_captured = df["mfe_captured"].dropna()

    # % Lost Good Trades
    mfe_threshold = df["mfe"].quantile(0.70)     # Soglia MFE: top 30% dei trade
    lost_good_mask = (df["return"] < 0) & (df["mfe"] > mfe_threshold)
    lost_good_pct = lost_good_mask.mean() * 100

    # Exit reasons
    exit_counts = df["exit_reason"].value_counts()
 
    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,

        # ritorni
        "Expectancy": returns.mean(),
        "Median_ret": returns.median(),
        "p25_ret": returns.quantile(0.25),
        "p75_ret": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "mean_pos_ret": positive_returns.mean(),
        "Median_pos_ret": positive_returns.median(),
        "p25_pos_ret": positive_returns.quantile(0.25),
        "p75_pos_ret": positive_returns.quantile(0.75),
        "Skew_pos_ret": positive_returns.skew(),
        "n_pos_trade": len(positive_returns),

        # ritorni negativi
        "mean_neg_ret": negative_retunrs.mean(),
        "n_neg_trade": len(negative_retunrs),

        # % MFE captured
        "%MFE_captured_median": mfe_captured.median() if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p25": mfe_captured.quantile(0.25) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p75": mfe_captured.quantile(0.75) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_mean": mfe_captured.mean() if len(mfe_captured) > 0 else np.nan,

        # % Lost Good Trades
        "%LostGood": lost_good_pct,

        # rischio
        "MAE_mean": df["mae"].mean(),
        "MAE_p75": df["mae"].quantile(0.75),
        "MAE_p90": df["mae"].quantile(0.90),

        "MFE_mean": df["mfe"].mean(),
        "MFE_p75": df["mfe"].quantile(0.75),
        "MFE_p90": df["mfe"].quantile(0.90),

        # edge dynamics
        "TMFE_mean": df["tmfe"].mean(),

        # Exit reasons
        "n_trailing_stop": exit_counts.get("trailing_stop", 0),
        "n_time_exit": exit_counts.get("time_exit", 0),
    }

    return metrics


# --- MAIN LOOP ---
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    for hp in holding_periods:
        for tp in trailing_levels:
            trades = collect_trades_trailing_low(df, hp, tp)
            metrics = compute_metrics(trades)

            if metrics is None:
                continue

            metrics["ticker"] = ticker
            metrics["HP"] = hp
            metrics["trailing_pct_%"] = tp*100
            results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Risultati su Low
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- ANALISI AGGREGATA (AVERAGE ACROSS TICKERS) ---
# Raggruppa per HP e trailing_pct e calcola le medie
agg_results = results_df.groupby(['HP', 'trailing_pct_%']).agg({
    "n_trades": 'sum',
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    "mean_pos_ret": "mean",
    "n_pos_trade": "sum",
    "mean_neg_ret": "mean",
    "n_neg_trade": "sum",
    'Median_ret': 'mean',
    #'p25_ret': 'mean',
    #'p75_ret': 'mean',
    'Skew_ret': 'mean',
    #'%MFE_captured_median': 'mean',
    #'%MFE_captured_p25': 'mean',
    #'%MFE_captured_p75': 'mean',
    '%MFE_captured_mean': 'mean',
    '%LostGood': 'mean',
    'MAE_mean': 'mean',
    #'MAE_p75': 'mean',
    #'MAE_p90': 'mean',
    'MFE_mean': 'mean',
    #'MFE_p75': 'mean',
    #'MFE_p90': 'mean',
    'TMFE_mean': 'mean',
    "n_trailing_stop": "sum",
    "n_time_exit": "sum",
}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)



----------
### Test su Close
----------

def Oops_di_Williams_Long(df):

    patterns = [None, None]

    # BODY STRENGTH
    df["Body"] = (df["close"] - df["open"]).abs()
    df["Range"] = df["high"] - df["low"]
    df["Body_to_Range"] = df["Body"] / df["Range"].replace(0, np.nan)

    body_class = []

    for ratio in df["Body_to_Range"]:
        if pd.isna(ratio):
            body_class.append(None)
        elif ratio >= 0.7:
            body_class.append("long body")
        elif ratio >= 0.4:
            body_class.append("average body")
        else:
            body_class.append("small body")

    df["body_class"] = body_class

    # RANGE QUANTILI
    df["Total_Range"] = df["high"] - df["low"]

    range_class = []

    for i in range(len(df)):
        if i < 100:
            range_class.append(None)
        else:
            recent = df["Total_Range"].iloc[i-100:i]
            q33 = np.nanpercentile(recent, 33)
            q66 = np.nanpercentile(recent, 66)
            tr = df["Total_Range"].iloc[i]
            if tr < q33:
                range_class.append("smaller-sized")
            elif tr > q66:
                range_class.append("larger-sized")
            else:
                range_class.append("average")
    
    df["Range_Class"] = range_class
    

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

        body_class = df["body_class"].iloc[i]
        body_class_minus1 = df["body_class"].iloc[i-1]
    
        range_class = df["Range_Class"].iloc[i]
        range_class_minus1 = df["Range_Class"].iloc[i-1]


# --- PATTERN ---
        # LONG con pressione ribassista media e forte
        if (
            # 1° candle
            range_class_minus1 in ["larger-sized","average"] and
            body_class_minus1 in ["long body", "average body"] and
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
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


def collect_trades_trailing_close(df, holding_period, trailing_pct):
    trades = []

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        closes = df["close"].iloc[i:i+holding_period-1]
        lows  = df["low"].iloc[i:i+holding_period-1]
        highs = df["high"].iloc[i:i+holding_period-1]

        max_price = entry_price
        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(len(closes)):
            current_high = highs.iloc[j]
            current_close = closes.iloc[j]

            # aggiorna massimo favorevole
            if current_high > max_price:
                max_price = current_high

            trailing_level = max_price * (1 - trailing_pct)

            # condizione di uscita
            if current_close < trailing_level:
                exit_price = current_close # attenzione: prendo come prezzo di uscita il close!!
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # se non è mai scattato il trailing → exit time-based
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"

        # Return
        ret = (exit_price - entry_price) / entry_price

        # Positive return
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # %MFE captured-mean
        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "trailing_pct": trailing_pct,
            "exit_bar": exit_index,
            "exit_reason": exit_reason,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
    returns = df["return"].dropna()
    positive_returns = df["positive_return"].dropna()
    negative_returns = df.loc[df["return"] < 0, "return"]

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    # % MFE captured
    mfe_captured = df["mfe_captured"].dropna()

    # % Lost Good Trades
    mfe_threshold = df["mfe"].quantile(0.70)     # Soglia MFE: top 30% dei trade
    lost_good_mask = (df["return"] < 0) & (df["mfe"] > mfe_threshold)
    lost_good_pct = lost_good_mask.mean() * 100

    # Exit reasons
    exit_counts = df["exit_reason"].value_counts()
 
    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,

        # ritorni
        "Expectancy": returns.mean(),
        "Median_ret": returns.median(),
        "p25_ret": returns.quantile(0.25),
        "p75_ret": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "mean_pos_ret": positive_returns.mean(),
        "Median_pos_ret": positive_returns.median(),
        "p25_pos_ret": positive_returns.quantile(0.25),
        "p75_pos_ret": positive_returns.quantile(0.75),
        "Skew_pos_ret": positive_returns.skew(),
        "n_pos_trade": len(positive_returns),

        # ritorni negativi
        "mean_neg_ret": negative_returns.mean(),
        "n_neg_trade": len(negative_returns),

        # % MFE captured
        "%MFE_captured_median": mfe_captured.median() if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p25": mfe_captured.quantile(0.25) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p75": mfe_captured.quantile(0.75) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_mean": mfe_captured.mean() if len(mfe_captured) > 0 else np.nan,

        # % Lost Good Trades
        "%LostGood": lost_good_pct,

        # rischio
        "MAE_mean": df["mae"].mean(),
        "MAE_p75": df["mae"].quantile(0.75),
        "MAE_p90": df["mae"].quantile(0.90),

        "MFE_mean": df["mfe"].mean(),
        "MFE_p75": df["mfe"].quantile(0.75),
        "MFE_p90": df["mfe"].quantile(0.90),

        # edge dynamics
        "TMFE_mean": df["tmfe"].mean(),

        # Exit reasons
        "n_trailing_stop": exit_counts.get("trailing_stop", 0),
        "n_time_exit": exit_counts.get("time_exit", 0),
    }

    return metrics


# --- MAIN LOOP ---
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    for hp in holding_periods:
        for tp in trailing_levels:
            trades = collect_trades_trailing_close(df, hp, tp)
            metrics = compute_metrics(trades)

            if metrics is None:
                continue

            metrics["ticker"] = ticker
            metrics["HP"] = hp
            metrics["trailing_pct_%"] = tp*100
            results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Risultati su Close

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- ANALISI AGGREGATA (AVERAGE ACROSS TICKERS) ---
# Raggruppa per HP e trailing_pct e calcola le medie
agg_results = results_df.groupby(['HP', 'trailing_pct_%']).agg({
    'n_trades': 'sum',
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    "mean_pos_ret": "mean",
    "n_pos_trade": "sum",
    "mean_neg_ret": "mean",
    "n_neg_trade": "sum",
    'Median_ret': 'mean',
    #'p25_ret': 'mean',
    #'p75_ret': 'mean',
    'Skew_ret': 'mean',
    #'%MFE_captured_median': 'mean',
    #'%MFE_captured_p25': 'mean',
    #'%MFE_captured_p75': 'mean',
    '%MFE_captured_mean': 'mean',
    '%LostGood': 'mean',
    'MAE_mean': 'mean',
    #'MAE_p75': 'mean',
    #'MAE_p90': 'mean',
    'MFE_mean': 'mean',
    #'MFE_p75': 'mean',
    #'MFE_p90': 'mean',
    'TMFE_mean': 'mean',
    "n_trailing_stop": "sum",
    "n_time_exit": "sum",
}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)


--------------
--------------
## Chandelier Exit (ATR-based)
--------------
--------------

### Parametri

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY
Timeframe_ATR = Resolution.DAILY

tol_dir = 0.2   # Tollereranza per candela direzionale = 20% 
min_body_ratio =  0.6 # Tollereranza per candela direzionale = body/ total range --> 60 % of body

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [3,4,5,6,7,10]

atr_periods = [5] #[5, 14]
chandelier_multipliers = [2.0, 2.5, 3.0, 3.5]


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

def Oops_di_Williams_Long(df):

    patterns = [None, None]

    # BODY STRENGTH
    df["Body"] = (df["close"] - df["open"]).abs()
    df["Range"] = df["high"] - df["low"]
    df["Body_to_Range"] = df["Body"] / df["Range"].replace(0, np.nan)

    body_class = []

    for ratio in df["Body_to_Range"]:
        if pd.isna(ratio):
            body_class.append(None)
        elif ratio >= 0.7:
            body_class.append("long body")
        elif ratio >= 0.4:
            body_class.append("average body")
        else:
            body_class.append("small body")

    df["body_class"] = body_class

    # RANGE QUANTILI
    df["Total_Range"] = df["high"] - df["low"]

    range_class = []

    for i in range(len(df)):
        if i < 100:
            range_class.append(None)
        else:
            recent = df["Total_Range"].iloc[i-100:i]
            q33 = np.nanpercentile(recent, 33)
            q66 = np.nanpercentile(recent, 66)
            tr = df["Total_Range"].iloc[i]
            if tr < q33:
                range_class.append("smaller-sized")
            elif tr > q66:
                range_class.append("larger-sized")
            else:
                range_class.append("average")
    
    df["Range_Class"] = range_class
    

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

        body_class = df["body_class"].iloc[i]
        body_class_minus1 = df["body_class"].iloc[i-1]
    
        range_class = df["Range_Class"].iloc[i]
        range_class_minus1 = df["Range_Class"].iloc[i-1]


# --- PATTERN ---
        # LONG con pressione ribassista media e forte
        if (
            # 1° candle
            range_class_minus1 in ["larger-sized","average"] and
            body_class_minus1 in ["long body", "average body"] and
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
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

# ATR
def compute_atr(df, symbol, atr_period):

    atr = qb.ATR(symbol, atr_period, MovingAverageType.Wilders, Timeframe_ATR)
    atr_values = []

    for time, row in df.iterrows():

        bar = TradeBar(
            time,
            symbol,
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"]
        )

        atr.Update(bar)
        atr_values.append(atr.Current.Value if atr.IsReady else np.nan)

    return atr_values


def collect_trades_chandelier(df, holding_period, atr_period, atr_mult):
    trades = []

    df = df.copy()
    df["ATR"] = compute_atr(df, symbol, atr_period)

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        closes  = df["close"].iloc[i:i+holding_period-1]
        highs  = df["high"].iloc[i:i+holding_period-1]
        lows   = df["low"].iloc[i:i+holding_period-1]
        atrs  = df["ATR"].iloc[i:i+holding_period-1]

        max_price = entry_price
        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(len(highs)):
            current_high = highs.iloc[j]
            current_low  = lows.iloc[j]
            current_atr  = atrs.iloc[j]
            current_close  = closes.iloc[j]

            if np.isnan(current_atr):
                continue

            if current_high > max_price:
                max_price = current_high

            chandelier_level = max_price - atr_mult * current_atr

            if current_close < chandelier_level:
                exit_price = current_close
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # Time exit se non scatta
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"


        # Return
        ret = (exit_price - entry_price) / entry_price

        # Positive return
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # %MFE captured-mean
        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "atr_period": atr_period,
            "atr_mult": atr_mult,
            "exit_bar": exit_index,
            "exit_reason": exit_reason,
            "stop_loss_level": (current_close - entry_price) / entry_price * 100,  # % distanza SL
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
    returns = df["return"].dropna()
    positive_returns = df["positive_return"].dropna()
    negative_returns = df.loc[df["return"] < 0, "return"]

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    # % MFE captured
    mfe_captured = df["mfe_captured"].dropna()

    # % Lost Good Trades
    mfe_threshold = df["mfe"].quantile(0.70)     # Soglia MFE: top 30% dei trade
    lost_good_mask = (df["return"] < 0) & (df["mfe"] > mfe_threshold)
    lost_good_pct = lost_good_mask.mean() * 100

    # Exit reasons
    exit_counts = df["exit_reason"].value_counts()
 
    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,

        # ritorni
        "Expectancy": returns.mean(),
        "Median_ret": returns.median(),
        "p25_ret": returns.quantile(0.25),
        "p75_ret": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "mean_pos_ret": positive_returns.mean(),
        "Median_pos_ret": positive_returns.median(),
        "p25_pos_ret": positive_returns.quantile(0.25),
        "p75_pos_ret": positive_returns.quantile(0.75),
        "Skew_pos_ret": positive_returns.skew(),
        "n_pos_trade": len(positive_returns),

        # ritorni negativi
        "mean_neg_ret": negative_returns.mean(),
        "n_neg_trade": len(negative_returns),

        # % MFE captured
        "%MFE_captured_median": mfe_captured.median() if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p25": mfe_captured.quantile(0.25) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_p75": mfe_captured.quantile(0.75) if len(mfe_captured) > 0 else np.nan,
        "%MFE_captured_mean": mfe_captured.mean() if len(mfe_captured) > 0 else np.nan,

        # % Lost Good Trades
        "%LostGood": lost_good_pct,

        # rischio
        "MAE_mean": df["mae"].mean(),
        "MAE_p75": df["mae"].quantile(0.75),
        "MAE_p90": df["mae"].quantile(0.90),

        "MFE_mean": df["mfe"].mean(),
        "MFE_p75": df["mfe"].quantile(0.75),
        "MFE_p90": df["mfe"].quantile(0.90),

        # edge dynamics
        "TMFE_mean": df["tmfe"].mean(),

        # Exit reasons
        "n_time_exit": exit_counts.get("time_exit", 0),
        "n_trailing_stop": exit_counts.get("trailing_stop",0),
        
        # Stop loss stats
        "avg_sl_distance_%": df["stop_loss_level"].mean(),
    }

    return metrics


# --- MAIN LOOP ---
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    for hp in holding_periods:
        for atr_p in atr_periods:
            for atr_m in chandelier_multipliers:

                trades = collect_trades_chandelier(df, hp, atr_p, atr_m)
                metrics = compute_metrics(trades)

                if metrics is None:
                    continue

                metrics["ticker"] = ticker
                metrics["HP"] = hp
                metrics["ATR_period"] = atr_p
                metrics["ATR_mult"] = atr_m

                results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Risultati
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Raggruppa per HP e trailing_pct e calcola le medie
agg_results = results_df.groupby(['HP', 'ATR_mult']).agg({
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    "mean_pos_ret": "mean",
    "mean_neg_ret": "mean",
    'Median_ret': 'mean',
    #'p25_ret': 'mean',
    #'p75_ret': 'mean',
    'Skew_ret': 'mean',
    #'%MFE_captured_median': 'mean',
    #'%MFE_captured_p25': 'mean',
    #'%MFE_captured_p75': 'mean',
    '%MFE_captured_mean': 'mean',
    '%LostGood': 'mean',
    'MAE_mean': 'mean',
    #'MAE_p75': 'mean',
    #'MAE_p90': 'mean',
    'MFE_mean': 'mean',
    #'MFE_p75': 'mean',
    #'MFE_p90': 'mean',
    'TMFE_mean': 'mean',
    'n_trades': 'sum',
    "n_trailing_stop": "sum",
    "n_time_exit": "sum",
    "avg_sl_distance_%":"mean",
}).reset_index()

agg_results = agg_results.round(2)
display(agg_results)



























