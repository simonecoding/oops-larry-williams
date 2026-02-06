--------------
## MAE Based Stop
--------------

### Parametri
# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY
Timeframe_ATR = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [3,4,5,6,7,10]

mae_percentiles = [0.75, 0.85, 0.95]
mfe_percentiles = {"P87.5": 0.875}
atr_periods = [5]
chandelier_multipliers = [2.5]


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


### su MFE + Chandelier
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


def compute_mfe_distribution(df, holding_period):
    mfe_list = []

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        highs = df["high"].iloc[i:i+holding_period -1]

        mfe = (highs.max() - entry_price) / entry_price

        if mfe > 0:
            mfe_list.append(mfe)

    return np.array(mfe_list)


def collect_trades_chandelier_and_mfe_target(df, holding_period, atr_period, atr_mult, mfe_target):
    trades = []

    df = df.copy()
    df["ATR"] = compute_atr(df, symbol, atr_period)

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        target_price = entry_price * (1 + mfe_target)

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
            
            # Aggiorna il massimo su cui calcolare chandelier
            if current_high > max_price:
                max_price = current_high

            chandelier_level = max_price - atr_mult * current_atr

            # PROFIT TARGET MFE-BASED
            if current_high >= target_price:
                exit_price = target_price
                exit_index = j
                exit_reason = "take_profit"
                break

            # Chandelier exit
            if current_close < chandelier_level:
                exit_price = current_close
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # Time exit
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
            "take_profit_level": (target_price - entry_price) / entry_price * 100,  # % distanza SL
        })

    return trades


def collect_trades_chandelier_and_mfe_target_and_MAE_STOP(df, holding_period, atr_period, atr_mult, mfe_target, mae_stop_pct, symbol):
    trades = []
    
    # Calcola ATR
    df = df.copy()
    df["ATR"] = compute_atr(df, symbol, atr_period)

    for i in range(1, len(df) - holding_period):
        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        target_price = entry_price * (1 + mfe_target)  # <-- AGGIUNTO

        opens  = df["open"].iloc[i:i+holding_period-1].reset_index(drop=True)
        closes = df["close"].iloc[i:i+holding_period-1].reset_index(drop=True)
        lows   = df["low"].iloc[i:i+holding_period-1].reset_index(drop=True)
        highs  = df["high"].iloc[i:i+holding_period-1].reset_index(drop=True)
        atrs   = df["ATR"].iloc[i:i+holding_period-1].reset_index(drop=True)  # <-- AGGIUNTO reset_index

        max_price = entry_price
        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(len(highs)):
            current_high = highs.iloc[j]
            current_low  = lows.iloc[j]
            current_atr  = atrs.iloc[j]
            current_close = closes.iloc[j]

            if np.isnan(current_atr):
                continue
            
            # Aggiorna il massimo su cui calcolare chandelier
            if current_high > max_price:
                max_price = current_high

            chandelier_level = max_price - atr_mult * current_atr

            # 1. PROFIT TARGET MFE-BASED (priorità più alta)
            if current_high >= target_price:
                exit_price = target_price
                exit_index = j
                exit_reason = "take_profit"
                break

            # 2. MAE STOP
            mae_j = abs((lows.iloc[:j+1].min() - entry_price) / entry_price)
            if mae_j >= mae_stop_pct:
                exit_price = entry_price * (1 - mae_stop_pct)
                exit_index = j
                exit_reason = "mae_stop"
                break

            # 3. Chandelier exit
            if current_close < chandelier_level:
                exit_price = current_close
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # 4. TIME EXIT
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"

        # Calcoli metriche
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        mae = abs((lows.iloc[:exit_index+1].min() - entry_price) / entry_price)
        mfe = abs((highs.iloc[:exit_index+1].max() - entry_price) / entry_price)

        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        tmfe = highs.iloc[:exit_index+1].idxmax() - highs.iloc[:exit_index+1].index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret * 100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "atr_period": atr_period,
            "atr_mult": atr_mult,
            "exit_bar": exit_index,
            "exit_reason": exit_reason,
            "mae_stop_pct": mae_stop_pct * 100,
            "stop_loss_level": (current_close - entry_price) / entry_price * 100,  # % distanza SL
            "take_profit_level": (target_price - entry_price) / entry_price * 100,
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

    # Exit time
    exit_time = df["exit_bar"].mean()
 
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
        "n_take_profit": exit_counts.get("take_profit", 0),
        
        # Stop loss stats
        "avg_sl_distance_%": df["stop_loss_level"].mean(),
        "avg_tp_distance_%": df["take_profit_level"].mean(),

        # Exit Time
        "Mean_HP_exit": exit_time
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
        # --- MFE DISTRIBUTION per ogni HP ---
        mfe_dist = compute_mfe_distribution(df, hp)
        if len(mfe_dist) == 0:
            continue

        mfe_targets = {
            name: np.percentile(mfe_dist, p * 100)
            for name, p in mfe_percentiles.items()
        }

        # Loop su ATR e multiplier
        for atr_period in atr_periods:
            for atr_mult in chandelier_multipliers:
                for mfe_label, mfe_target in mfe_targets.items():
                    
                    # ---------- BASELINE (no MAE stop) ----------
                    base_trades = collect_trades_chandelier_and_mfe_target(
                        df, hp, atr_period, atr_mult, mfe_target
                    )
                    base_df = pd.DataFrame(base_trades)

                    if base_df.empty:
                        continue

                    # Calcola MAE percentiles dalla baseline
                    mae_levels = {
                        p: base_df["mae"].quantile(p) / 100
                        for p in mae_percentiles
                    }

                    # ---------- TEST con MAE STOP ----------
                    for p, mae_stop in mae_levels.items():
                        trades = collect_trades_chandelier_and_mfe_target_and_MAE_STOP(
                            df, hp, atr_period, atr_mult, mfe_target, mae_stop, symbol
                        )

                        metrics = compute_metrics(trades)
                        if metrics is None:
                            continue

                        metrics.update({
                            "ticker": ticker,
                            "HP": hp,
                            "ATR_period": atr_period,
                            "ATR_mult": atr_mult,
                            "MFE_target": mfe_label,
                            "MAE_stop_percentile": p,
                            "MAE_stop_value_%": mae_stop * 100,
                            "exit_type": "MFE+Chandelier+MAE_Stop"
                        })

                        results.append(metrics)

results_df = pd.DataFrame(results)
print("ok")



#### Results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Raggruppa per HP e trailing_pct e calcola le medie
agg_results = results_df.groupby(['HP', "MAE_stop_percentile"]).agg({
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
    "n_take_profit": "sum",
    "n_time_exit": "sum",
    "avg_tp_distance_%":"mean",
    "Mean_HP_exit": "mean"
}).reset_index()

agg_results = agg_results.round(2)

display(agg_results)



--------------------
## Stop su minimo del giorno di ingresso + (-) ATR
--------------------

### Parametri
# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY
Timeframe_ATR = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [3,4,5,6,7,10]

mfe_percentiles = {"P87.5": 0.875}
atr_periods = [5] # sia per chandelier che per stop
chandelier_multipliers = [2.5]
stoploss_multipliers = [2,4,6]


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

tickers = Indici_US_ETF


### Su MFE + Chandelier
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


def compute_mfe_distribution(df, holding_period):
    mfe_list = []

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        highs = df["high"].iloc[i:i+holding_period -1]

        mfe = (highs.max() - entry_price) / entry_price

        if mfe > 0:
            mfe_list.append(mfe)

    return np.array(mfe_list)


def collect_trades_chandelier_and_mfe_target_and_stop_price(df, holding_period, atr_period, atr_mult, mfe_target, stoploss_multiplier):
    trades = []

    # Calcola ATR
    df = df.copy()
    df["ATR"] = compute_atr(df, symbol, atr_period)

    for i in range(1, len(df) - holding_period):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]
        target_price = entry_price * (1 + mfe_target)

        # Stop Loss
        atr_before_entry = df["ATR"].iloc[i-1]  # ATR del giorno prima dell'ingresso
        if np.isnan(atr_before_entry):
            continue

        stop_loss = entry_price - (stoploss_multiplier * atr_before_entry) # <-- larghezza ATR X stop

        opens  = df["open"].iloc[i:i+holding_period-1].reset_index(drop=True)
        closes = df["close"].iloc[i:i+holding_period-1].reset_index(drop=True)
        lows   = df["low"].iloc[i:i+holding_period-1].reset_index(drop=True)
        highs  = df["high"].iloc[i:i+holding_period-1].reset_index(drop=True)
        atrs   = df["ATR"].iloc[i:i+holding_period-1].reset_index(drop=True) 

        max_price = entry_price
        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(len(highs)):
            current_high = highs.iloc[j]
            current_low  = lows.iloc[j]
            current_atr  = atrs.iloc[j]
            current_close = closes.iloc[j]

            if np.isnan(current_atr):
                continue

            # Aggiorna il massimo su cui calcolare chandelier
            if current_high > max_price:
                max_price = current_high

            chandelier_level = max_price - atr_mult * current_atr

            # 1. Verifica STOP LOSS
            if current_low <= stop_loss:
                exit_price = stop_loss
                exit_index = j
                exit_reason = "stop_loss"
                break

            # 2. PROFIT TARGET MFE-BASED
            if current_high >= target_price:
                exit_price = target_price
                exit_index = j
                exit_reason = "take_profit"
                break

            # 3. Chandelier exit
            if current_close < chandelier_level:
                exit_price = current_close
                exit_index = j
                exit_reason = "trailing_stop"
                break

        # 4. TIME EXIT
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"

        # Calcoli metriche
        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        mae = abs((lows.iloc[:exit_index+1].min() - entry_price) / entry_price)
        mfe = abs((highs.iloc[:exit_index+1].max() - entry_price) / entry_price)

        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        tmfe = highs.iloc[:exit_index+1].idxmax() - highs.iloc[:exit_index+1].index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret * 100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "atr_period": atr_period,
            "atr_mult": atr_mult,
            "exit_bar": exit_index,
            "exit_reason": exit_reason,
            "SL_level": (stop_loss - entry_price) / entry_price * 100,  # % distanza SL
            "SL_Chandelier": (current_close - entry_price) / entry_price * 100, 
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
    mfe_threshold = df["mfe"].quantile(0.70)
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
        "mean_neg_ret": negative_returns.mean() if len(negative_returns) > 0 else np.nan,
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
        "n_stop_loss": exit_counts.get("stop_loss", 0),
        "n_trailing_stop": exit_counts.get("trailing_stop", 0),
        "n_time_exit": exit_counts.get("time_exit", 0),
        
        # Stop loss stats
        "avg_sl_level_dist_%": df["SL_level"].mean(),
        "avg_sl_chandelier_dist_%": df["SL_Chandelier"].mean(),
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
        # --- MFE DISTRIBUTION per ogni HP ---
        mfe_dist = compute_mfe_distribution(df, hp)

        if len(mfe_dist) == 0:
            continue

        mfe_targets = {
            name: np.percentile(mfe_dist, p * 100)
            for name, p in mfe_percentiles.items()
        }

        # Ora loop sui MFE target
        for label, mfe_target in mfe_targets.items():
            for atr_p in atr_periods:
                for atr_m in chandelier_multipliers:
                    for sl_mult in stoploss_multipliers:  # ✅ NUOVO LOOP

                        trades = collect_trades_chandelier_and_mfe_target_and_stop_price(
                            df, hp, atr_p, atr_m, mfe_target, sl_mult  # ✅ PASSA sl_mult
                        )
                        metrics = compute_metrics(trades)

                        if metrics is None:
                            continue

                        metrics["ticker"] = ticker
                        metrics["HP"] = hp
                        metrics["MFE_target"] = label
                        metrics["trailing_type"] = "MFE_percentile_TimeFail"
                        metrics["ATR_period"] = atr_p
                        metrics["ATR_mult"] = atr_m
                        metrics["SL_mult"] = sl_mult  # ✅ SALVA IL MULTIPLIER

                        results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


#### Risultati
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP', "SL_mult"]).agg({
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
    "avg_sl_level_dist_%": "mean",
    "n_stop_loss": "sum",
    "avg_sl_chandelier_dist_%": "mean",
    "n_trailing_stop": "sum",
    "n_time_exit": "sum",
}).reset_index()


agg_results = agg_results.round(2)
display(agg_results)





----------------------
## Larry Setup
----------------------

### Parametri
# --- PARAMETRI ---
start_date = datetime(2005, 1, 1)
end_date   = datetime(2025, 12, 18)

timeframe_security = Resolution.DAILY
Timeframe_ATR = Resolution.DAILY

toll_gap = 0.000   # esempio 0.002 = 0.2%

holding_periods = [1,2,3,4,5,6,7,10]

atr_periods = [5] # sia per chandelier che per stop
stoploss_multipliers = [2,4,6]


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


def collect_trades_LW_open_profit_exit(df, holding_period, atr_period, stoploss_multiplier):
    trades = []

    # Calcola ATR
    df = df.copy()
    df["ATR"] = compute_atr(df, symbol, atr_period)

    for i in range(1, len(df) - holding_period -1):

        if df["Pattern"].iloc[i] not in PATTERN_DIRECTION:
            continue

        entry_price = df["low"].iloc[i-1]

        # Stop Loss
        atr_before_entry = df["ATR"].iloc[i-1]  # ATR del giorno prima dell'ingresso
        if np.isnan(atr_before_entry):
            continue

        stop_loss = entry_price - (stoploss_multiplier * atr_before_entry) # <-- larghezza ATR X stop

        opens = df["open"].iloc[i:i+holding_period].reset_index(drop=True)
        closes = df["close"].iloc[i:i+holding_period].reset_index(drop=True)
        lows =  df["low"].iloc[i:i+holding_period].reset_index(drop=True)

        exit_price = None
        exit_index = None
        exit_reason = None

        for j in range(1, len(opens)):

            # 1. Check if open price is in profit
            if opens.iloc[j] >= entry_price:
                exit_price = opens.iloc[j]
                exit_index = j
                exit_reason = "larry_exit"
                break

            # 2. Verifica STOP LOSS
            if lows.iloc[j] <= stop_loss:
                exit_price = stop_loss
                exit_index = j
                exit_reason = "stop_loss"
                break

        # 3. Time exit se non esce prima
        if exit_price is None:
            exit_price = closes.iloc[-1]
            exit_index = len(closes) - 1
            exit_reason = "time_exit"

        ret = (exit_price - entry_price) / entry_price
        positive_ret = ret if ret > 0 else np.nan

        mae = abs((df["low"].iloc[i:i+exit_index+1].min() - entry_price) / entry_price)
        mfe = abs((df["high"].iloc[i:i+exit_index+1].max() - entry_price) / entry_price)

        mfe_captured = (positive_ret / mfe) if mfe > 0 else np.nan

        tmfe = df["high"].iloc[i:i+exit_index+1].idxmax() - df["high"].iloc[i:i+exit_index+1].index[0]
        tmfe = tmfe.days if hasattr(tmfe, "days") else int(tmfe)

        trades.append({
            "return": ret * 100,
            "positive_return": positive_ret * 100 if not np.isnan(positive_ret) else np.nan,
            "mae": mae * 100,
            "mfe": mfe * 100,
            "mfe_captured": mfe_captured * 100 if not np.isnan(mfe_captured) else np.nan,
            "tmfe": tmfe,
            "HP": holding_period,
            "exit_bar": exit_index,
            "exit_reason": exit_reason,
            "SL_level": (stop_loss - entry_price) / entry_price * 100,  # % distanza SL
            "larry_exit_lev": (exit_price - entry_price) / entry_price * 100,  # % distanza SL
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

    # Exit Time
    exit_bars = df["exit_bar"].mean()

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
        "n_larry_exit": exit_counts.get("larry_exit", 0),
        "n_time_exit": exit_counts.get("time_exit", 0),
        "n_stop_loss": exit_counts.get("stop_loss", 0),

        # Exit time
        "Mean_HP_of_exit": exit_bars,

        # Stop loss stats
        "avg_sl_level_dist_%": df["SL_level"].mean(),
        "avg_larry_exit_dist_%": df["larry_exit_lev"].mean(),
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
            for sl_mult in stoploss_multipliers:
                trades = collect_trades_LW_open_profit_exit(df, hp, atr_p, sl_mult)
                metrics = compute_metrics(trades)

                if metrics is None:
                    continue

                metrics["ticker"] = ticker
                metrics["HP"] = hp
                metrics["trailing_type"] = "LW_Open_Profit_Exit"
                metrics["ATR_period"] = atr_p
                metrics["SL_mult"] = sl_mult
                results.append(metrics)

results_df = pd.DataFrame(results)


print("ok")


### Risultati
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

agg_results = results_df.groupby(['HP', "SL_mult"]).agg({
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
    'n_trades': 'sum',
    "n_larry_exit": "sum",
    "avg_larry_exit_dist_%": "mean",
    "n_time_exit": "sum",
    "avg_sl_level_dist_%": "mean",
    "n_stop_loss": "sum",
    "Mean_HP_of_exit": "mean",
}).reset_index()


agg_results = agg_results.round(2)
display(agg_results)

