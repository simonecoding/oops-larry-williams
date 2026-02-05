## Micro regime analysis - only long


### Parametri

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

tol_dir = 0.2   # Tollereranza per candela direzionale = 20% 
min_body_ratio =  0.6 # Tollereranza per candela direzionale = body/ total range --> 60 % of body

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



--------------
### prev_candle_directional - Gap>0
--------------


def Oops_di_Williams_Long(df):

    patterns = [None,None]

    for i in range(2, len(df)):

        o, h, l, c = map(float, df.iloc[i][["open","high","low","close"]])
        o_minus1, h_minus1, l_minus1, c_minus1 = map(float, df.iloc[i-1][["open","high","low","close"]])

        body_minus1 = abs(c_minus1 - o_minus1)
        total_range_minus1 = h_minus1 - l_minus1

        candle_directional_minus1 = (
            (h_minus1 - o_minus1) / (h_minus1 - l_minus1 + 1e-9) < tol_dir and
            (c_minus1 - l_minus1) / (h_minus1 - l_minus1 + 1e-9) < tol_dir and
            body_minus1 >= min_body_ratio*total_range_minus1 and
            c_minus1 < o_minus1  
        )

    # --- PATTERN long ---
        if (
            # 1° candle (candela direzionale ribassista)
            candle_directional_minus1 and
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

# --- collect trades ---
def collect_trades(df, hp_to_analyze):

    trades = []

    for i in range(1, len(df) - hp_to_analyze - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        highs = df["high"].iloc[i:i + hp_to_analyze -1] # -1 because I count the entry date as day 1
        lows  = df["low"].iloc[i:i + hp_to_analyze -1]
        closes = df["close"].iloc[i:i + hp_to_analyze -1]

        exit_price = closes.iloc[-1]

        # Return
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": hp_to_analyze,
            "trade_date": entry_date,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
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
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        print(f"Nessun dato per {ticker}")
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    trades = collect_trades(df, hp_to_analyze)
    metrics = compute_metrics(trades)

    if metrics is None:
        continue

    metrics["ticker"] = ticker
    metrics["HP"] = hp_to_analyze

    results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP']).agg({
    "n_trades": "sum",
    'Expectancy': 'mean',
    "Median_ret_pct": "mean",
    "Skew_ret": "mean",
    'Median_pos_ret_pct': 'mean',
    'MAE_mean_pct': 'mean',
    'MAE_p75_pct': 'mean',
    'MFE_mean_pct': 'mean',
    'MFE_p75_pct': 'mean',
    'TMFE_mean': 'mean',

}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)


--------------
--------------
# Range & quantili
--------------
--------------


## Parametri & Lista Tickers

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

toll_gap = 0.00   # esempio 0.002 = 0.2%

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


----------
## Strong pressure
----------

# FUNZIONE Oops di Williams
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
        # LONG con pressione ribassista forte
        if (
            # 1° candle (candela ribassista forte)
            range_class_minus1 in ["larger-sized"] and    # , "smaller-sized" ,"average"
            body_class_minus1 in ["long body"] and  # , "small body" , "average body"
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
            o < l_minus1 * (1 - toll_gap) and
            h >= l_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono il low precedente
        ):
            patterns.append("Long con pressione ribassista forte")
        
        else:
            patterns.append(None)

    df['Pattern'] = patterns
    return df


# --- MAPPA LONG/SHORT ---
PATTERN_DIRECTION = {
    # long
    "Long con pressione ribassista forte": "LONG",
}


# --- Funzione collect trades ---
def collect_trades(df, hp_to_analyze):

    trades = []

    for i in range(1, len(df) - hp_to_analyze - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        highs = df["high"].iloc[i:i + hp_to_analyze -1] # -1 because I count the entry date as day 1
        lows  = df["low"].iloc[i:i + hp_to_analyze -1]
        closes = df["close"].iloc[i:i + hp_to_analyze -1]

        exit_price = closes.iloc[-1]

        # Return
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": hp_to_analyze,
            "trade_date": entry_date,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
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
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        print(f"Nessun dato per {ticker}")
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    trades = collect_trades(df, hp_to_analyze)
    metrics = compute_metrics(trades)

    if metrics is None:
        continue

    metrics["ticker"] = ticker
    metrics["HP"] = hp_to_analyze

    results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP']).agg({
    "n_trades": "sum",
    'Expectancy': 'mean',
    "Median_ret_pct": "mean",
    "Skew_ret": "mean",
    'Median_pos_ret_pct': 'mean',
    'MAE_mean_pct': 'mean',
    'MAE_p75_pct': 'mean',
    'MFE_mean_pct': 'mean',
    'MFE_p75_pct': 'mean',
    'TMFE_mean': 'mean',

}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)


-----------
## Intermediate Pressure 
-----------
# FUNZIONE Oops di Williams
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
        # LONG con pressione ribassista forte
        if (
            # 1° candle (candela ribassista forte)
            range_class_minus1 in ["larger-sized","average"] and    # , "smaller-sized" ,"average"
            body_class_minus1 in ["long body", "average body"] and  # , "small body" , "average body"
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
            o < l_minus1 * (1 - toll_gap) and
            h >= l_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono il low precedente
        ):
            patterns.append("Long con pressione ribassista forte")
        
        else:
            patterns.append(None)

    df['Pattern'] = patterns
    return df


# --- MAPPA LONG/SHORT ---
PATTERN_DIRECTION = {
    # long
    "Long con pressione ribassista forte": "LONG",
}


# --- Funzione collect trades ---
def collect_trades(df, hp_to_analyze):

    trades = []

    for i in range(1, len(df) - hp_to_analyze - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        highs = df["high"].iloc[i:i + hp_to_analyze -1] # -1 because I count the entry date as day 1
        lows  = df["low"].iloc[i:i + hp_to_analyze -1]
        closes = df["close"].iloc[i:i + hp_to_analyze -1]

        exit_price = closes.iloc[-1]

        # Return
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        # MAE / MFE
        mae = abs((lows.min() - entry_price) / entry_price)
        mfe = abs((highs.max() - entry_price) / entry_price)

        # TMFE
        tmfe = highs.idxmax() - highs.index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": hp_to_analyze,
            "trade_date": entry_date,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
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
results = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        print(f"Nessun dato per {ticker}")
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    trades = collect_trades(df, hp_to_analyze)
    metrics = compute_metrics(trades)

    if metrics is None:
        continue

    metrics["ticker"] = ticker
    metrics["HP"] = hp_to_analyze

    results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")



#### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP']).agg({
    "n_trades": "sum",
    'Expectancy': 'mean',
    "Median_ret_pct": "mean",
    "Skew_ret": "mean",
    'Median_pos_ret_pct': 'mean',
    'MAE_mean_pct': 'mean',
    'MAE_p75_pct': 'mean',
    'MFE_mean_pct': 'mean',
    'MFE_p75_pct': 'mean',
    'TMFE_mean': 'mean',

}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)



---------
---------
# Local trend up VS Local trend down
---------
---------

## Parametri

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
    # Local Trend
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma5_slope"] = df["sma5"] - df["sma5"].shift(5)
    

    trades = collect_trades(df, hp_to_analyze)
    all_trades.extend(trades)

trades_df = pd.DataFrame(all_trades)

# --- Parte I micro ----
df_trend_up = trades_df[trades_df["sma5_slope"] > 0]
df_trend_down = trades_df[trades_df["sma5_slope"] < 0]


# --- Parte II micro ---
metrics_trend_up = compute_metrics_from_df(df_trend_up)
metrics_trend_down = compute_metrics_from_df(df_trend_down)



print("ok")


#### Results Local trend up VS Local trend down

summary_context = pd.DataFrame([
    {"Regime": "TrendUp",**metrics_trend_up},
    {"Regime": "TrendDown",**metrics_trend_down}
])

display(summary_context.round(2))


-------------
-------------
# GAP
-------------
-------------

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

ATR_PERIOD = 14
Timeframe_ATR = Resolution.DAILY
margine_sl = 0.1

holding_periods = [7]

# --- DEFINIZIONE BUCKET GAP ---
gap_buckets = {
    'G0': (0.0, 0.002),      # 0 - 0.2%
    'G1': (0.002, 0.005),    # 0.2 - 0.5%
    'G2': (0.005, 0.01),     # 0.5 - 1.0%
    'G3': (0.01, 999)        # > 1.0%
}

# --- LISTA TICKER ---
Indici_US_ETF = ["SPY", "QQQ", "IWM", "DIA"]
tickers = Indici_US_ETF


def Oops_di_Williams_Long_with_Gap(df):
    """
    Identifica pattern LONG e calcola il gap_ratio per ogni pattern
    """
    patterns = [None, None]
    gap_ratios = [None, None]
    
    
    for i in range(2, len(df)):
        o, h, l, c = map(float, df.iloc[i][["open", "high", "low", "close"]])
        o_minus1, h_minus1, l_minus1, c_minus1 = map(float, df.iloc[i-1][["open", "high", "low", "close"]])
        
        # CALCOLO GAP RATIO (LONG)
        gap_ratio = (o - l_minus1) / l_minus1
        
        # PATTERN LONG
        if (
            o < l_minus1 and         # gap down
            h >= l_minus1            # ritorno al low precedente
        ):
            patterns.append("Oops di williams Long")
            gap_ratios.append(gap_ratio)

        else:
            patterns.append(None)
            gap_ratios.append(None)
    
    df['Pattern'] = patterns
    df['Gap_Ratio'] = gap_ratios
    return df


def assign_gap_bucket(gap_ratio, buckets):
    """
    Assegna un trade al bucket corretto in base al gap_ratio
    """
    if gap_ratio is None or np.isnan(gap_ratio):
        return None
    
    for bucket_name, (min_gap, max_gap) in buckets.items():
        if min_gap <= abs(gap_ratio) < max_gap:
            return bucket_name
    
    return None


def test_entries_with_gap(df, holding_period, gap_buckets):
    """
    Testa le entrate e classifica per bucket di gap
    """
    trades_data = []
    
    for i in range(1, len(df) - holding_period - 1):
        pattern = df["Pattern"].iloc[i]
        
        if pattern != "Oops di williams Long":   # !!
            continue
        
        gap_ratio = df["Gap_Ratio"].iloc[i]
        bucket = assign_gap_bucket(gap_ratio, gap_buckets)
        
        if bucket is None:
            continue
        
        entry_price = df["low"].iloc[i-1]
        exit_price = df["close"].iloc[i + holding_period-1]
        
        # MAE e MFE per LONG
        h_slice = df["high"].iloc[i:i+holding_period]
        l_slice = df["low"].iloc[i:i+holding_period]
        
        mae = abs((l_slice.min() - entry_price) / entry_price)
        mfe = abs((h_slice.max() - entry_price) / entry_price)
        
        ret = (exit_price - entry_price) / entry_price
        
        trades_data.append({
            "return": ret,
            "mae": mae,
            "mfe": mfe,
            "gap_ratio": gap_ratio,
            "gap_bucket": bucket,
            "holding_period": holding_period
        })
    
    return trades_data


def compute_expectancy(trades_list):
    """
    Calcola l'expectancy
    """
    if len(trades_list) == 0:
        return np.nan
    
    returns = np.array([t["return"] for t in trades_list])
    
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    p_win = len(wins) / len(returns)
    p_loss = 1 - p_win
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    expectancy = p_win * avg_win - p_loss * avg_loss
    
    return expectancy * 100


# --- MAIN ANALYSIS ---
print("="*80)
print("ANALISI GAP - OOPS DI WILLIAMS LONG")
print("="*80)

all_trades = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)
    
    if history.empty:
        continue
    
    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long_with_Gap(df)
    df["Ticker"] = ticker
    
    for hp in holding_periods:
        trades_data = test_entries_with_gap(df, hp, gap_buckets)
        
        for trade in trades_data:
            trade["ticker"] = ticker
        
        all_trades.extend(trades_data)


# --- ANALISI PER BUCKET E HOLDING PERIOD ---
df_all_trades = pd.DataFrame(all_trades)

print(f"\nTotale trade analizzati: {len(df_all_trades)}\n")

# Struttura risultati
results = []

for bucket in ['G0', 'G1', 'G2', 'G3']:
    for hp in holding_periods:
        # Filtra trade per bucket e holding period
        df_filtered = df_all_trades[
            (df_all_trades['gap_bucket'] == bucket) &
            (df_all_trades['holding_period'] == hp)
        ]
        
        if len(df_filtered) == 0:
            continue
        
        # Calcola metriche
        n_trades = len(df_filtered)

        win_rate = (df_filtered['return'] > 0).sum() / n_trades * 100

        expectancy = compute_expectancy(df_filtered.to_dict('records'))

        median_ret_pct = df_filtered['return'].median() * 100

        skew_ret = df_filtered['return'].skew()

        Median_pos_ret_pct = df_filtered.loc[df_filtered['return'] > 0, 'return'].median() * 100

        mae_pct = df_filtered['mae'].mean() * 100
        mfe_pct = df_filtered['mfe'].mean() * 100
        mae_p75_pct = df_filtered['mae'].quantile(0.75) * 100
        mfe_p75_pct = df_filtered['mfe'].quantile(0.75) * 100


        results.append({
            'Gap_Bucket': bucket,
            'Holding': hp,
            'N_Trades': n_trades,
            'Expectancy': expectancy,
            "Median_ret_pct": median_ret_pct,
            "Skew_ret": skew_ret,
            "Median_pos_ret_pct": Median_pos_ret_pct,
            "MAE_mean_pct":mae_pct,
            'MAE_p75_pct': mae_p75_pct,
            "MFE_mean_pct":mfe_pct,
            'MFE_p75_pct': mfe_p75_pct,
        })

# Crea DataFrame risultati
df_results = pd.DataFrame(results)
df_results = df_results.round(2)

# --- TABELLE PER BUCKET ---
for bucket in ['G0', 'G1', 'G2', 'G3']:
    df_bucket = df_results[df_results['Gap_Bucket'] == bucket].copy()
    
    if len(df_bucket) == 0:
        continue
    
    df_bucket = df_bucket.set_index('Holding')
    df_bucket = df_bucket.drop('Gap_Bucket', axis=1)
    
    bucket_range = gap_buckets[bucket]
    title = f"{bucket}: Gap {bucket_range[0]*100:.1f}% - {bucket_range[1]*100:.1f}%"
    
    display(HTML(f"<h3>{title}</h3>"))
    display(df_bucket)

print("\n✅ Analisi completata!")


