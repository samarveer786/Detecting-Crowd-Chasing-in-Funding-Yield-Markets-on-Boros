#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# In[12]:


funding_df=pd.read_csv("binance_btc_funding_daily_2020.csv")
funding_df['fundingAPR']=funding_df['daily_funding_rate']*365
funding_df.head()


# In[79]:


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


# In[80]:


implied_df=simulate_implied_apr(funding_df, maturities=(30,90,180), alpha=0.6)
#implied_df.head()


# In[81]:


implied_df.columns = implied_df.columns.astype(str)

implied_df['30']=implied_df['30']/100
implied_df['90']=implied_df['90']/100
implied_df['180']=implied_df['180']/100


# In[82]:


# funding_df already has DatetimeIndex
funding_df.index = pd.to_datetime(funding_df.index)

implied_df = implied_df.copy()



merged = funding_df.join(implied_df)
merged = merged.reset_index(drop=True)

merged["date"] = pd.to_datetime(merged["date"])


merged = merged.reset_index(drop=True)



# In[83]:


merged = merged.rename(columns={
    "30": "impliedAPR_30d",
    "90": "impliedAPR_90d",
    "180": "impliedAPR_180d"
})
#merged.head()


# In[84]:


merged['gap_30']=merged['impliedAPR_30d']-merged['fundingAPR']
merged['gap_90']=merged['impliedAPR_90d']-merged['fundingAPR']
merged['gap_180']=merged['impliedAPR_180d']-merged['fundingAPR']


# In[85]:


T = 15   # days horizon proxy
S= 45

merged["gap_30_mean"] = merged['gap_30'].ewm(span=T, min_periods=T).mean()
merged["gap_30_std"] = merged['gap_30'].rolling(S, min_periods=S).std()
merged['gap_30_zscore']=(merged['gap_30']-merged["gap_30_mean"])/merged["gap_30_std"]


merged.tail()


# In[182]:


merged['fundingAPR'].max()
merged['impliedAPR_90d'].max()


# In[86]:


T = 15   # days horizon proxy
S= 45

merged["gap_90_mean"] = merged['gap_90'].ewm(span=T, min_periods=T).mean()
merged["gap_90_std"] = merged['gap_90'].rolling(S, min_periods=S).std()
merged['gap_90_zscore']=(merged['gap_90']-merged["gap_90_mean"])/merged["gap_90_std"]


#merged.tail()


# In[87]:


T = 15   # days horizon proxy
S= 45

merged["gap_180_mean"] = merged['gap_180'].ewm(span=T, min_periods=T).mean()
merged["gap_180_std"] = merged['gap_180'].rolling(S, min_periods=S).std()
merged['gap_180_zscore']=(merged['gap_180']-merged["gap_180_mean"])/merged["gap_180_std"]

#merged.tail()


# In[179]:


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


# In[168]:


def backtest_yu_gap_pro(df, capital=10000, entry_z=1.0, exit_z=0.25, alloc_pct=0.25):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365

    # --- Normalize APR units ---
    for c in ["fundingAPR", "impliedAPR_90d"]:
        if df[c].abs().median() > 1:
            df[c] = df[c] / 100.0

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df) - 1):

        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        z_today = today["gap_90_zscore"]
        ma3 = today["realized_ma3"]

        # ===== ENTRY SNAPSHOT (t+1 execution) =====
        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["impliedAPR_90d"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (realized_entry > implied_entry * 1.5)

        # ===== ENTRY =====
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

        # ===== ACTIVE TRADE =====
        if open_trade and i >= open_trade["entry_idx"]:

            realized_today = today["fundingAPR"]

            if open_trade["position"] == 1:
                pnl = open_trade["notional"] * (realized_today - open_trade["entry_implied"]) * dt
            else:
                pnl = open_trade["notional"] * (open_trade["entry_implied"] - realized_today) * dt

            open_trade["trade_pnl"] += pnl
            equity += pnl
            free_equity += pnl

            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2
            days_held = i - open_trade["entry_idx"]

            # ===== EXIT =====
            if abs(z_today) < exit_z or shock or days_held >= 90:

                trades.append({
                    "side": open_trade["side"],
                    "entry_idx": open_trade["entry_idx"],
                    "exit_idx": i,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": open_trade["entry_implied"],
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": today["impliedAPR_90d"],
                    "exit_realized": realized_today,
                    "days_held": days_held,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%": (pnl / (capital * days_held / 365)) *100 if days_held > 0 else np.nan
                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades), df


# In[169]:


trades_df_90, equity_df_90=backtest_yu_gap_pro(
        merged,
        capital=10000,
        entry_z=1,
        exit_z=0.01,
        alloc_pct=1
    )
trades_df_90


# In[170]:


print(trades_df_90['ROI_%'].sum())


# In[171]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# make sure index is datetime
merged.index = pd.to_datetime(merged['date'])

plt.figure(figsize=(12,6))

plt.plot(merged.index, equity_df_90["equity"], label="equity", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Equity Curve : Maturity = 90d")
plt.xlabel("Date")
plt.ylabel("Equity")

# format x-axis as dates
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[172]:


trades_df_90["side"].value_counts()


# In[173]:


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


# In[174]:


metrics = boros_metrics(trades_df_90, equity_df_90)
metrics


# In[175]:


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


# In[176]:


df_stress = stress_funding_environment(merged)

trades_stress, equity_stress = backtest_yu_gap_pro(df_stress)

trades_stress


# In[177]:


metrics_stress = boros_metrics(trades_stress, equity_stress)
metrics_stress


# In[ ]:





# In[ ]:




