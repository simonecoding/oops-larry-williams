Test Exit


------------
## Estensione HP a 25
------------

### Parametri & Lista Tickers

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

toll_gap = 0.00   # esempio 0.002 = 0.2%

holding_periods = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]


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


### Intermediate Pressure 
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
        # LONG
        if (
            # 1° candle
            range_class_minus1 in ["larger-sized","average"] and    # , "smaller-sized" ,"average"
            body_class_minus1 in ["long body", "average body"] and  # , "small body" , "average body"
            c_minus1 < o_minus1 and     # candela ribassista

            # 2° candle
            o < l_minus1 * (1 - toll_gap) and
            h >= l_minus1 # aggiunto questo per considerare i segnali che effettivamente raggiungono il low precedente
        ):
            patterns.append("Long con pressione ribassista media")
        
        else:
            patterns.append(None)

    df['Pattern'] = patterns
    return df


# --- MAPPA LONG/SHORT ---
PATTERN_DIRECTION = {
    # long
    "Long con pressione ribassista media": "LONG",
}


# --- Funzione collect trades ---
def collect_trades(df, holding_period):

    trades = []

    for i in range(1, len(df) - holding_period):

        pattern = df["Pattern"].iloc[i]
        if pattern not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        entry_date = df.index[i-1]  # Prendo la data della riga i-1 come data di entrata

        # GESTIONE SPECIALE PER HP = 1
        if holding_period == 1:
            # Con hp=1, esci subito alla chiusura della candela di entrata
            exit_price = df["close"].iloc[i]
            mae = abs((df["low"].iloc[i] - entry_price) / entry_price)
            mfe = abs((df["high"].iloc[i] - entry_price) / entry_price)
            tmfe = 0  # stesso giorno
        else:
            # HP >= 2
            highs = df["high"].iloc[i:i + holding_period - 1]
            lows  = df["low"].iloc[i:i + holding_period - 1]
            closes = df["close"].iloc[i:i + holding_period - 1]
            
            exit_price = closes.iloc[-1]
            mae = abs((lows.min() - entry_price) / entry_price)
            mfe = abs((highs.max() - entry_price) / entry_price)
            tmfe = highs.idxmax() - highs.index[0]
            tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        # Return
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan
        negative_ret = ret if ret <0 else np.nan


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "negative_ret_pct": negative_ret*100 if not np.isnan(negative_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": holding_period,
            "trade_date": entry_date,
        })

    return trades


def compute_metrics(trades):
    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)
    returns = df["return_pct"].dropna()
    positive_returns = df["positive_return_pct"].dropna()
    negative_returns = df["negative_ret_pct"].dropna()

    metrics = {
        "n_trades": len(df),

        "Expectancy": returns.mean(),

        "Win_rate_pct": (len(positive_returns) / len(returns))*100,

        # ritorni
        "Median_ret_pct": returns.median(),
        "p25_ret_pct": returns.quantile(0.25),
        "p75_ret_pct": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "mean_pos_ret_pct":positive_returns.mean(),
        "Median_pos_ret_pct": positive_returns.median(),
        #"p25_pos_ret": positive_returns.quantile(0.25),
        #"p75_pos_ret": positive_returns.quantile(0.75),
        #"Skew_pos_ret": positive_returns.skew(),

        # negative returns
        "mean_neg_ret_pct": negative_returns.mean(),

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

    for hp in holding_periods:
        trades = collect_trades(df, hp)
        metrics = compute_metrics(trades)

        if metrics is None:
            continue

        metrics["ticker"] = ticker
        metrics["HP"] = hp

        results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP']).agg({
    "n_trades": "sum",
    'Expectancy': 'mean',
    "Win_rate_pct": "mean",
    "Median_ret_pct": "mean",
    "p25_ret_pct": "mean",
    "p75_ret_pct": "mean",
    "Skew_ret": "mean",
    'Median_pos_ret_pct': 'mean',
    #"mean_neg_ret_pct": "mean",
    'MAE_mean_pct': 'mean',
    'MAE_p75_pct': 'mean',
    'MFE_mean_pct': 'mean',
    'MFE_p75_pct': 'mean',
    'TMFE_mean': 'mean'
}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)


#### Grafici

# Expectancy
plt.figure(figsize=(12, 6))
plt.plot(agg_results['HP'], agg_results['Expectancy'], marker='o', linewidth=2, markersize=8)
plt.xlabel('HP', fontsize=12)
plt.ylabel('Expectancy', fontsize=12)
plt.title('Expectancy per HP', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# win rate
plt.figure(figsize=(12, 6))
plt.plot(agg_results['HP'], agg_results['Win_rate_pct'], marker='o', linewidth=2, markersize=8)
plt.xlabel('HP', fontsize=12)
plt.ylabel('Win_rate_pct', fontsize=12)
plt.title('Win_rate_pct per HP', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



-------------
# Edge Structure
-------------


## Parametri

# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [3,4,5,6,7,10]


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


## Close su High

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


def collect_trades(df, holding_period):
    trades = []

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        # GESTIONE SPECIALE PER HP = 1
        if holding_period == 1:
            # Con hp=1, esci subito alla chiusura della candela di entrata
            exit_price = df["high"].iloc[i]
            mae = abs((df["low"].iloc[i] - entry_price) / entry_price)
            mfe = abs((df["high"].iloc[i] - entry_price) / entry_price)
            tmfe = 0  # stesso giorno
        else:
            # HP >= 2
            highs = df["high"].iloc[i:i + holding_period - 1]
            lows  = df["low"].iloc[i:i + holding_period - 1]
            closes = df["high"].iloc[i:i + holding_period - 1]
            
            exit_price = closes.iloc[-1]
            mae = abs((lows.min() - entry_price) / entry_price)
            mfe = abs((highs.max() - entry_price) / entry_price)
            tmfe = highs.idxmax() - highs.index[0]
            tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        # Return
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan
        #negative_ret = ret if ret <0 else np.nan


        trades.append({
            "return_pct": ret * 100,
            "positive_return_pct": positive_ret*100 if not np.isnan(positive_ret) else np.nan,
            "mae_pct": mae * 100,
            "mfe_pct": mfe * 100,
            "tmfe": tmfe,
            "HP": holding_period
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

        "Win_rate_pct": (len(positive_returns) / len(returns))*100,

        # ritorni
        "Median_ret_pct": returns.median(),
        "p25_ret_pct": returns.quantile(0.25),
        "p75_ret_pct": returns.quantile(0.75),
        "Skew_ret": returns.skew(),

        # ritorni positivi
        "mean_pos_ret_pct":positive_returns.mean(),
        "Median_pos_ret_pct": positive_returns.median(),
        #"p25_pos_ret_pct": positive_returns.quantile(0.25),
        #"p75_pos_ret_pct": positive_returns.quantile(0.75),
        #"Skew_pos_ret": positive_returns.skew(),

        # negative returns
        #"mean_neg_ret_pct": negative_returns.mean(),

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
all_trades = []

for ticker in tickers:
    symbol = qb.add_equity(ticker).symbol
    history = qb.history(symbol, start_date, end_date, timeframe_security)

    if history.empty:
        continue

    df = history.loc[symbol].copy()
    df = Oops_di_Williams_Long(df)

    for hp in holding_periods:
        trades = collect_trades(df, hp)
        all_trades.extend(trades)

        metrics = compute_metrics(trades)

        if metrics is None:
            continue

        metrics["ticker"] = ticker
        metrics["HP"] = hp
        results.append(metrics)

results_df = pd.DataFrame(results)
trades_df = pd.DataFrame(all_trades)


print("ok")



### results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP']).agg({
    "n_trades": "sum",
    'Expectancy': 'mean',
    "Win_rate_pct": "mean",
    "Median_ret_pct": "mean",
    "p25_ret_pct": "mean",
    "p75_ret_pct": "mean",
    "Skew_ret": "mean",
    'Median_pos_ret_pct': 'mean',
    'MAE_mean_pct': 'mean',
    'MAE_p75_pct': 'mean',
    'MFE_mean_pct': 'mean',
    'MFE_p75_pct': 'mean',
    'TMFE_mean': 'mean'
}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)






























