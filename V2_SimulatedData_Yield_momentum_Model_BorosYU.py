#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# In[25]:


funding_df=pd.read_csv("binance_btc_funding_daily_2020.csv")
funding_df['fundingAPR']=funding_df['daily_funding_rate']*365
funding_df.head()


# In[26]:


import numpy as np
import pandas as pd

def simulate_implied_apr(funding_df, maturities, alpha=0.6):

    n = len(funding_df)
    exp_f = funding_df['fundingAPR'].ewm(span=14).mean()

    # 3-state risk premium regime
    states = ["risk_on", "neutral", "risk_off"]
    trans = [[0.9,0.1,0.0],[0.05,0.9,0.05],[0.0,0.1,0.9]]
    rp_vals = {"risk_on":3.5, "neutral":0.0, "risk_off":-2.0}

    state = "neutral"
    rp = []

    for _ in range(n):
        state = np.random.choice(states, p=trans[states.index(state)])
        rp.append(rp_vals[state])

    rp = pd.Series(rp, index=funding_df.index)

    implied = {}

    for T in maturities:
        lp = alpha * np.sqrt(T)
        implied[T] = exp_f + rp + lp + np.random.normal(0,0.25,n)

    return pd.DataFrame(implied)


# In[27]:


implied_df=simulate_implied_apr(funding_df, maturities=(30,90,180), alpha=0.6)
implied_df.head()


# In[28]:


implied_df.columns = implied_df.columns.astype(str)

implied_df['30']=implied_df['30']/100
implied_df['90']=implied_df['90']/100
implied_df['180']=implied_df['180']/100
implied_df.head()


# In[31]:


funding_df = funding_df.copy()
implied_df = implied_df.copy()

funding_df.index = pd.to_datetime(funding_df.index)
implied_df.index = pd.to_datetime(implied_df.index)

# time-aligned join
merged = funding_df.join(implied_df, how="inner").sort_index()

# keep DatetimeIndex – do NOT reset index
merged.head()


# In[32]:


merged = merged.copy()

merged["date"] = pd.to_datetime(merged["date"])
merged.set_index("date", inplace=True)

# remove fake epoch index name
merged.index.name = None


# In[33]:


merged = merged.rename(columns={
    "30": "impliedAPR_30d",
    "90": "impliedAPR_90d",
    "180": "impliedAPR_180d"
})
merged.head()


# In[34]:


merged['gap_30']=merged['impliedAPR_30d']-merged['fundingAPR']
merged['gap_90']=merged['impliedAPR_90d']-merged['fundingAPR']
merged['gap_180']=merged['impliedAPR_180d']-merged['fundingAPR']


# In[35]:


T = 15   # days horizon proxy
S= 45

merged["gap_30_mean"] = merged['gap_30'].ewm(span=T, min_periods=T).mean()
merged["gap_30_std"] = merged['gap_30'].rolling(S, min_periods=S).std()
merged['gap_30_zscore']=(merged['gap_30']-merged["gap_30_mean"])/merged["gap_30_std"]


merged.tail()


# In[38]:


T = 15   # days horizon proxy
S= 45

merged["gap_90_mean"] = merged['gap_90'].ewm(span=T, min_periods=T).mean()
merged["gap_90_std"] = merged['gap_90'].rolling(S, min_periods=S).std()
merged['gap_90_zscore']=(merged['gap_90']-merged["gap_90_mean"])/merged["gap_90_std"]


#merged.tail()


# In[40]:


T = 15   # days horizon proxy
S= 45

merged["gap_180_mean"] = merged['gap_180'].ewm(span=T, min_periods=T).mean()
merged["gap_180_std"] = merged['gap_180'].rolling(S, min_periods=S).std()
merged['gap_180_zscore']=(merged['gap_180']-merged["gap_180_mean"])/merged["gap_180_std"]

#merged.tail()


# In[41]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

#plt.plot(merged.index, merged["impliedAPR_30d"], label="impliedAPR_30d", linewidth=2)
plt.plot(merged.index, merged["impliedAPR_90d"], label="impliedAPR_90d", linewidth=2)
#plt.plot(merged.index, merged["impliedAPR_180d"], label="impliedAPR_180d", linewidth=2)
#plt.plot(merged.index, merged['gap_30'], label="gap_30", linewidth=2)
#plt.plot(merged.index, merged['gap_90'], label="gap_90", linewidth=2)
#plt.plot(merged.index, merged['gap_180'], label="gap_180", linewidth=2)
#plt.plot(merged.index, merged['gap_30_zscore'], label="Gap_30d_Zscore", linewidth=2)
#plt.plot(merged.index, merged['gap_90_zscore'], label="Gap_90d_Zscore", linewidth=2)
#plt.plot(merged.index, merged['gap_180_zscore'], label="Gap_180d_Zscore", linewidth=2)




plt.plot(merged.index, merged["fundingAPR"], label="Funding", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Comparison: Implied APR(90d) Vs. Funding APR ")
plt.xlabel("Date")
plt.ylabel("APR (%)")
plt.tight_layout()
plt.show()


# In[77]:


def backtest_yu_gap_pro(df, capital=10000, entry_z=1.0, exit_z=0.25, alloc_pct=0.25):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365
    MATURITY_DAYS = 90

    for c in ["fundingAPR", "impliedAPR_30d"]:
        if df[c].abs().median() > 1:
            df[c] /= 100.0

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df) - 1):

        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        z_today = today["gap_30_zscore"]
        ma3 = today["realized_ma3"]

        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["impliedAPR_30d"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (realized_entry > implied_entry * 1.5)

        if open_trade is None and not np.isnan(z_today):

            notional = free_equity * alloc_pct

            if z_today < -entry_z and not block_long:
                open_trade = {
                    "side": "LONG_YU",
                    "position": 1,
                    "entry_idx": i + 1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

            elif z_today > entry_z and not block_short:
                open_trade = {
                    "side": "SHORT_YU",
                    "position": -1,
                    "entry_idx": i + 1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

        if open_trade and i >= open_trade["entry_idx"]:

            I0 = open_trade["entry_implied"]
            realized_today = today["fundingAPR"]

            cashflow = open_trade["notional"] * (
                (realized_today - I0) if open_trade["position"] == 1 else (I0 - realized_today)
            ) * dt

            open_trade["trade_pnl"] += cashflow
            equity += cashflow
            free_equity += cashflow

            days_held = i - open_trade["entry_idx"]
            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2
            curve_blowoff = (
                open_trade["side"] == "SHORT_YU" and
                today["impliedAPR_30d"] > open_trade["entry_implied"] * 1.25
            )

            
            if abs(z_today) < exit_z or shock or curve_blowoff or days_held >= MATURITY_DAYS:

                I_exit = today["impliedAPR_30d"]
                remaining_days = max(0, MATURITY_DAYS - days_held)

                repricing_pnl = open_trade["notional"] * (I0 - I_exit) * (remaining_days / 365)

                open_trade["trade_pnl"] += repricing_pnl
                equity += repricing_pnl
                free_equity += repricing_pnl

                trades.append({
                    "side": open_trade["side"],
                    "entry_idx": open_trade["entry_idx"],
                    "exit_idx": i,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": I0,
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": I_exit,
                    "exit_realized": realized_today,
                    "days_held": days_held,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%": (open_trade["trade_pnl"] / (open_trade["margin"]* days_held / 365) * 100 if days_held > 0 else np.nan)
                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades), df


# In[78]:


trades_df_90, equity_df_90= backtest_yu_gap_pro(merged, capital=10000, entry_z=1.0, exit_z=0.01, alloc_pct=1)
trades_df_90


# In[79]:


def boros_metrics(trades_df, equity_df, capital=10000):

    import numpy as np
    import pandas as pd

    df = trades_df.copy()
    pnl = df["trade_pnl"]

    # ---------- Core Carry Metrics ----------
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    Carry_Risk_Ratio = np.inf if len(losses)==0 else wins.mean() / abs(losses.mean())
    Tail_Loss_Freq   = (pnl < pnl.quantile(0.05)).mean()
    Worst_5pct_PnL   = pnl.quantile(0.05)
    Payoff_Skew      = pnl.skew()

    # ---------- Equity Curve ----------
    eq = equity_df["equity"].dropna()
    ret = eq.pct_change().dropna()

    cummax = eq.cummax()
    dd = eq / cummax - 1
    MaxDD = dd.min()

    Time_Underwater = (dd < 0).mean()

    # ---------- Crowding Hit Rate ----------
    crowded = df["entry_z"] > df["entry_z"].quantile(0.8)
    Crowding_HitRate = (df[crowded]["trade_pnl"] > 0).mean()

    # ---------- ROI & CAGR ----------
    total_pnl = pnl.sum()
    total_days = trades_df["days_held"].sum()

    ROI = (total_pnl / (capital )) * 100 if total_days > 0 else np.nan


    #days = (eq.index[-1] - eq.index[0]).days
    CAGR = (1 + total_pnl / capital) ** (1 / 6) - 1 if total_days > 0 else np.nan

    # ---------- Risk Ratios ----------
    rf_daily = 0.04 / 365
    Sharpe = (ret.mean() - rf_daily) / ret.std() * np.sqrt(365) if ret.std() > 0 else np.nan
    Calmar = (CAGR / abs(MaxDD)) if MaxDD != 0 else np.nan

    return {
        "Carry_Risk_Ratio": round(Carry_Risk_Ratio,2),
        "Tail_Loss_Freq": round(Tail_Loss_Freq,3),
        "Worst_5pct_PnL": round(Worst_5pct_PnL,2),
        "Payoff_Skew": round(Payoff_Skew,2),
        "Time_Underwater": round(Time_Underwater,2),
        "Crowding_HitRate": round(Crowding_HitRate,2),
        "ROI_%": round(ROI,2),
        "CAGR_%": round(CAGR*100,2),
        "Sharpe": round(Sharpe,2),
        "Calmar": round(Calmar,2),
        "MaxDD_%": round(MaxDD*100,2),
        "Total_Trades": len(df),
        "Losing_Trades": int((pnl < 0).sum())
    }


# In[80]:


metrics = boros_metrics(trades_df_90, equity_df_90)
metrics


# In[81]:


def stress_funding_environment(df,
                               jump_prob=0.015,
                               jump_scale=6,
                               fat_tail_df=3,
                               liquidity_shock_prob=0.01):

    import numpy as np
    import pandas as pd
    from scipy.stats import t

    df = df.copy()

    # ----- 1. Funding jump shocks -----
    jumps = np.random.rand(len(df)) < jump_prob
    jump_sizes = np.random.choice([1,-1], len(df)) * np.random.exponential(jump_scale, len(df))
    df.loc[jumps, "fundingAPR"] *= (1 + jump_sizes[jumps] / 100)

    # ----- 2. Fat-tailed micro noise -----
    df["fundingAPR"] += t.rvs(df=fat_tail_df, size=len(df)) * 0.002

    # ----- 3. Liquidity convexity regime -----
    shock = np.random.rand(len(df)) < liquidity_shock_prob
    df.loc[shock, "impliedAPR_90d"] *= np.random.uniform(1.5, 3.5, shock.sum())

    return df


# In[82]:


df_stress = stress_funding_environment(merged)

trades_stress, equity_stress = backtest_yu_gap_pro(df_stress)

trades_stress


# In[83]:


metrics_stress = boros_metrics(trades_stress, equity_stress)
metrics_stress


# In[ ]:





# In[ ]:




