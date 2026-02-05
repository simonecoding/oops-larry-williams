# Long & Short entry classica su low


## Parametri & Lista Tickers

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


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

---------------
## Long
---------------

def Oops_di_Williams_Long_Short(df):

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
        # LONG con pressione ribassista forte
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


# --- Funzione per testare le entrate ---
def test_entries(df, holding_period):
    """
    Restituisce una lista di dizionari con tutti i dettagli dei trade
    """

    trades_data = []

    for i in range(1, len(df) - holding_period - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        side = PATTERN_DIRECTION[pattern]

        if pattern == "Oops di williams Long":
            entry_price = df["low"].iloc[i-1]    # entry sul ritorno del minimo precedente
            exit_price  = df["close"].iloc[i + holding_period-1]

        ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
    
        trades_data.append({
            "return": ret,
            "side": side,
            "holding_period": holding_period
        })

    return trades_data


def compute_metrics(trades_data):
    if len(trades_data) == 0:
        return None

    df = pd.DataFrame(trades_data)
    returns = df["return"].dropna()
    print(returns)
    positive_returns = df.loc[df["return"] > 0, "return"].dropna()
    negative_retunrs = df.loc[df["return"] < 0, "return"].dropna()

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,
        "Expectancy": returns.mean(),

        # ritorni
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
        "Median_neg_ret": negative_retunrs.median(),
        "p25_neg_ret": negative_retunrs.quantile(0.25),
        "p75_neg_ret": negative_retunrs.quantile(0.75),
        "Skew_neg_ret": negative_retunrs.skew(),
        "n_neg_trade": len(negative_retunrs),
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
    df = Oops_di_Williams_Long_Short(df)

    for hp in holding_periods:
        trades_data = test_entries(df, hp)
        metrics = compute_metrics(trades_data)

        if metrics is None:
            continue

        metrics["ticker"] = ticker
        metrics["HP"] = hp
        results.append(metrics)

results_df = pd.DataFrame(results)

print("ok")


### Results Long
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Raggruppa per HP e calcola le medie
agg_results = results_df.groupby(['HP']).agg({
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    #'Median_ret': 'mean',
    #'p25_ret': 'mean',
    #'p75_ret': 'mean',
    #'Skew_ret': 'mean',
    'n_trades': 'sum',
    "mean_pos_ret": "mean",
    #'Median_pos_ret': 'mean',
    #'p25_pos_ret': 'mean',
    #'p75_pos_ret': 'mean',
    #'Skew_pos_ret': 'mean',
    'n_pos_trade': 'sum',
    "mean_neg_ret": "mean",
    #'Median_neg_ret': 'mean',
    #'p25_neg_ret': 'mean',
    #'p75_neg_ret': 'mean',
    #'Skew_neg_ret': 'mean',
    'n_neg_trade': 'sum',
}).reset_index()

agg_results = agg_results.round(2)
display(agg_results)


---------------
## Short
---------------

def Oops_di_Williams_Long_Short(df):

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


# --- Funzione per testare le entrate ---
def test_entries(df, holding_period):
    """
    Restituisce una lista di dizionari con tutti i dettagli dei trade
    """

    trades_data = []

    for i in range(1, len(df) - holding_period - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        side = PATTERN_DIRECTION[pattern]

        if pattern == "Oops di williams Short":
            entry_price = df["high"].iloc[i-1]    # entry sul ritorno del massimo precedente
            exit_price  = df["close"].iloc[i + holding_period -1] # -1 alla fine per considerare dati intraday

        ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
    
        trades_data.append({
            "return": ret,
            "side": side,
            "holding_period": holding_period
        })

    return trades_data


def compute_metrics(trades_data):
    if len(trades_data) == 0:
        return None

    df = pd.DataFrame(trades_data)
    returns = df["return"].dropna()
    positive_returns = df.loc[df["return"] > 0, "return"].dropna()
    negative_retunrs = df.loc[df["return"] < 0, "return"].dropna()

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,
        "Expectancy": returns.mean(),

        # ritorni
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
        "Median_neg_ret": negative_retunrs.median(),
        "p25_neg_ret": negative_retunrs.quantile(0.25),
        "p75_neg_ret": negative_retunrs.quantile(0.75),
        "Skew_neg_ret": negative_retunrs.skew(),
        "n_neg_trade": len(negative_retunrs),
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
    df = Oops_di_Williams_Long_Short(df)

    for hp in holding_periods:
        trades_data = test_entries(df, hp)
        metrics = compute_metrics(trades_data)

        if metrics is None:
            continue

        metrics["ticker"] = ticker
        metrics["HP"] = hp
        results.append(metrics)

results_df = pd.DataFrame(results)

print("ok")


### Results short
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Raggruppa per HP e calcola le medie
agg_results = results_df.groupby(['HP']).agg({
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    #'Median_ret': 'mean',
    #'p25_ret': 'mean',
    #'p75_ret': 'mean',
    #'Skew_ret': 'mean',
    'n_trades': 'sum',
    "mean_pos_ret": "mean",
    #'Median_pos_ret': 'mean',
    #'p25_pos_ret': 'mean',
    #'p75_pos_ret': 'mean',
    #'Skew_pos_ret': 'mean',
    'n_pos_trade': 'sum',
    "mean_neg_ret": "mean",
    #'Median_neg_ret': 'mean',
    #'p25_neg_ret': 'mean',
    #'p75_neg_ret': 'mean',
    #'Skew_neg_ret': 'mean',
    'n_neg_trade': 'sum',
}).reset_index()

agg_results = agg_results.round(2)
display(agg_results)


------------
## LONG + SHORT
------------

def Oops_di_Williams_Long_Short(df):

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
        # LONG con pressione ribassista forte
        if (
            o < l_minus1 * (1 - toll_gap) and
            h >= l_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono il low precedente
        ):
            patterns.append("Oops di williams Long")
        
        elif (
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
    # long
    "Oops di williams Long": "LONG",
    # short
    "Oops di williams Short": "SHORT",
}


# --- Funzione per testare le entrate ---
def test_entries(df, holding_period):
    """
    Restituisce una lista di dizionari con tutti i dettagli dei trade
    """

    trades_data = []

    for i in range(1, len(df) - holding_period - 1):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        side = PATTERN_DIRECTION[pattern]

        if pattern == "Oops di williams Long":
            entry_price = df["low"].iloc[i-1]    # entry sul ritorno del minimo precedente
            exit_price  = df["close"].iloc[i + holding_period-1]

        elif pattern == "Oops di williams Short":
            entry_price = df["high"].iloc[i-1]    # entry sul ritorno del massimo precedente
            exit_price  = df["close"].iloc[i + holding_period -1] # -1 alla fine per considerare dati intraday

        ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
    
        trades_data.append({
            "return": ret,
            "side": side,
            "holding_period": holding_period
        })

    return trades_data


def compute_metrics(trades_data):
    if len(trades_data) == 0:
        return None

    df = pd.DataFrame(trades_data)
    returns = df["return"].dropna()
    positive_returns = df.loc[df["return"] > 0, "return"].dropna()
    negative_retunrs = df.loc[df["return"] < 0, "return"].dropna()

    # Win Rate
    win_rate = (df["return"] > 0).mean() * 100

    metrics = {
        "n_trades": len(df),

        "Win_rate_%": win_rate,
        "Expectancy": returns.mean(),

        # ritorni
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
        "Median_neg_ret": negative_retunrs.median(),
        "p25_neg_ret": negative_retunrs.quantile(0.25),
        "p75_neg_ret": negative_retunrs.quantile(0.75),
        "Skew_neg_ret": negative_retunrs.skew(),
        "n_neg_trade": len(negative_retunrs),
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
    df = Oops_di_Williams_Long_Short(df)

    for hp in holding_periods:
        trades_data = test_entries(df, hp)
        metrics = compute_metrics(trades_data)

        if metrics is None:
            continue

        metrics["ticker"] = ticker
        metrics["HP"] = hp
        results.append(metrics)

results_df = pd.DataFrame(results)

print("ok")


### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Raggruppa per HP e calcola le medie
agg_results = results_df.groupby(['HP']).agg({
    'Win_rate_%': 'mean',
    'Expectancy': 'mean',
    'Median_ret': 'mean',
    'p25_ret': 'mean',
    'p75_ret': 'mean',
    'Skew_ret': 'mean',
    'n_trades': 'sum',
    #"mean_pos_ret": "mean",
    #'Median_pos_ret': 'mean',
    #'p25_pos_ret': 'mean',
    #'p75_pos_ret': 'mean',
    #'Skew_pos_ret': 'mean',
    #'n_pos_trade': 'sum',
    #"mean_neg_ret": "mean",
    #'Median_neg_ret': 'mean',
    #'p25_neg_ret': 'mean',
    #'p75_neg_ret': 'mean',
    #'Skew_neg_ret': 'mean',
    #'n_neg_trade': 'sum',
}).reset_index()

agg_results = agg_results.round(2)
display(agg_results)


