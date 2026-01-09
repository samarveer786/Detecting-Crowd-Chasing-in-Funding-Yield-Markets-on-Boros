#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# In[3]:


implied_df = pd.read_csv("BTC_Boros_Daily_Implied_WDATE.csv")
implied_df.head()


# In[4]:


implied_df["date"] = pd.to_datetime(implied_df["hour"], utc=True).dt.tz_convert(None)
implied_df = implied_df.drop(columns=["hour"])
implied_df = implied_df.set_index("date").sort_index()


# In[5]:


funding_df=pd.read_csv("binance_btc_funding_daily_2020.csv")
funding_df['fundingAPR']=funding_df['daily_funding_rate']*365
funding_df.head()


# In[6]:


funding_df["date"] = pd.to_datetime(funding_df["date"], errors="coerce")
funding_df = funding_df.set_index("date").sort_index()


# In[7]:


start, end = pd.Timestamp("2025-07-27"), pd.Timestamp("2025-12-25")

implied_df = implied_df.loc[start:end]
funding_df = funding_df.loc[start:end]


# In[8]:


merged = implied_df.join(funding_df, how="inner")

merged.head()


# In[12]:


#del(merged['daily_funding_rate'])
#merged['0x0ee61be6f8f73fb1412cc1205fb706ea441901ab']=merged['0x0ee61be6f8f73fb1412cc1205fb706ea441901ab']/100
#merged['0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f']= merged['0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f']/100
#merged['0x6e51252c47e6f3f246a6f501ac059d7d24a5a286']= merged['0x6e51252c47e6f3f246a6f501ac059d7d24a5a286']/100
#merged['0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8']= merged['0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8']/100


# In[23]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(merged.index, merged["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"], label="1", linewidth=2)
#plt.plot(merged.index, merged["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"], label="2", linewidth=2)
#plt.plot(merged.index, merged["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"], label="3", linewidth=2)
#plt.plot(merged.index, merged["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"], label="Implied", linewidth=2)
#plt.plot(merged.index, merged["gap_zscore_4"], label="", linewidth=2)



plt.plot(merged.index, merged["fundingAPR"], label="Funding", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Z_Score_Gap")
plt.xlabel("Date")
plt.ylabel("Zscore")
plt.tight_layout()
plt.show()


# In[17]:


merged['gap<>4']=merged['0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8']-merged['fundingAPR']
T = 10   # days horizon proxy
S= 30

merged["gap_MA_4"] = merged['gap<>4'].ewm(span=T, min_periods=T).mean()
merged["gap_std_4"] = merged['gap<>4'].rolling(S, min_periods=S).std()
merged['gap_zscore_4']=(merged['gap<>4']-merged["gap_MA_4"])/merged["gap_std_4"]
#merged.tail(40)


# In[58]:


def backtest_yu_gap_pro_4(
        df,
        capital=10000,
        entry_z=1.0,
        exit_z=0,
        alloc_pct=0.25
    ):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df)-1):

        today = df.iloc[i]
        tomorrow = df.iloc[i+1]

        ts_today = df.index[i]
        ts_t1    = df.index[i+1]

        z_today = today["gap_zscore_4"]
        ma3     = today["realized_ma3"]

        # ========== ENTRY SNAPSHOT ==========
        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (
            (realized_entry > implied_entry*1.5) 
        )

        # ========== ENTRY ==========
        if open_trade is None and not np.isnan(z_today):

            notional = free_equity * alloc_pct

            if z_today < -entry_z and not block_long:
                open_trade = {
                    "side": "LONG_YU",
                    "position": 1,
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
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
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

        # ========== ACTIVE TRADE ==========
        if open_trade and i >= open_trade["entry_idx"]:

            realized_today = today["fundingAPR"]

            if open_trade["position"] == 1:
                pnl = open_trade["notional"] * (realized_today - open_trade["entry_implied"]) * dt
            else:
                pnl = open_trade["notional"] * (open_trade["entry_implied"] - realized_today) * dt

            open_trade["trade_pnl"] += pnl
            equity += pnl
            free_equity += pnl    # funding loss reduces free collateral

            # ---------- STOPLOSS (Funding shock) ----------
            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2

            # ---------- EXIT ----------
            days_held = (ts_today - open_trade["entry_date"]).days

            if abs(z_today) < exit_z or shock or days_held >= 90:


                trades.append({
                    "side": open_trade["side"],
                    "entry_date": open_trade["entry_date"],
                    "exit_date": ts_today,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": open_trade["entry_implied"],
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": today["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"],
                    "exit_realized": realized_today,
                    "days_held": (ts_today - open_trade["entry_date"]).days,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%":(pnl / (capital * days_held / 365)) * 100 if days_held > 0 else np.nan

                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades),df


# In[59]:


trades_df4, equity_df4 = backtest_yu_gap_pro_4(
        merged,
        capital=10000,
        entry_z=1.0,
        exit_z=0.05,
        alloc_pct=1
    )
trades_df4


# In[60]:


trades_df4['ROI_%'].sum()


# In[61]:


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

    ROI = (total_pnl / (capital * total_days / 365)) * 100 if total_days > 0 else np.nan


    days = (eq.index[-1] - eq.index[0]).days
    CAGR = (1 + total_pnl / capital) ** (365 / total_days) - 1 if total_days > 0 else np.nan

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


# In[62]:


metrics = boros_metrics(trades_df4, equity_df4)
metrics


# In[350]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

#plt.plot(merged.index, merged["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"], label="1", linewidth=2)
#plt.plot(merged.index, merged["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"], label="2", linewidth=2)
plt.plot(merged.index, merged["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"], label="3", linewidth=2)
#plt.plot(merged.index, merged["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"], label="4", linewidth=2)
#plt.plot(merged.index, merged["gap_zscore_3"], label="gap_zscore_3", linewidth=2)



#plt.plot(merged.index, merged["fundingAPR"], label="Funding", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Comparison_APR — Implied")
plt.xlabel("Date")
plt.ylabel("APR (%)")
plt.tight_layout()
plt.show()


# In[105]:


merged['gap<>3']=merged['0x6e51252c47e6f3f246a6f501ac059d7d24a5a286']-merged['fundingAPR']
T = 10   # days horizon proxy
S= 30

merged["gap_MA_3"] = merged['gap<>3'].ewm(span=T, min_periods=T).mean()
merged["gap_std_3"] = merged['gap<>3'].rolling(S, min_periods=S).std()
merged['gap_zscore_3']=(merged['gap<>3']-merged["gap_MA_3"])/merged["gap_std_3"]
#merged.tail()


# In[106]:


def backtest_yu_gap_pro_3(
        df,
        capital=10000,
        entry_z=1.0,
        exit_z=0,
        alloc_pct=0.25
    ):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df)-1):

        today = df.iloc[i]
        tomorrow = df.iloc[i+1]

        ts_today = df.index[i]
        ts_t1    = df.index[i+1]

        z_today = today["gap_zscore_3"]
        ma3     = today["realized_ma3"]

        # ========== ENTRY SNAPSHOT ==========
        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (
            (realized_entry > implied_entry*1.5) 
        )

        # ========== ENTRY ==========
        if open_trade is None and not np.isnan(z_today):

            notional = free_equity * alloc_pct

            if z_today < -entry_z and not block_long:
                open_trade = {
                    "side": "LONG_YU",
                    "position": 1,
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
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
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

        # ========== ACTIVE TRADE ==========
        if open_trade and i >= open_trade["entry_idx"]:

            realized_today = today["fundingAPR"]

            if open_trade["position"] == 1:
                pnl = open_trade["notional"] * (realized_today - open_trade["entry_implied"]) * dt
            else:
                pnl = open_trade["notional"] * (open_trade["entry_implied"] - realized_today) * dt

            open_trade["trade_pnl"] += pnl
            equity += pnl
            free_equity += pnl    # funding loss reduces free collateral

            # ---------- STOPLOSS (Funding shock) ----------
            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2

            # ---------- EXIT ----------
            days_held = (ts_today - open_trade["entry_date"]).days

            if abs(z_today) < exit_z or shock or days_held >= 90:


                trades.append({
                    "side": open_trade["side"],
                    "entry_date": open_trade["entry_date"],
                    "exit_date": ts_today,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": open_trade["entry_implied"],
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": today["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"],
                    "exit_realized": realized_today,
                    "days_held": (ts_today - open_trade["entry_date"]).days,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%":(pnl / (capital * days_held / 365)) * 100 if days_held > 0 else np.nan

                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades),df


# In[107]:


trades_df3, equity_df3 = backtest_yu_gap_pro_3(
        merged,
        capital=10000,
        entry_z=1.0,
        exit_z=0.01,
        alloc_pct=0.5
    )
trades_df3


# In[108]:


trades_df3['ROI_%'].sum()


# In[114]:


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

    ROI = (total_pnl / (capital * total_days / 365)) * 100 if total_days > 0 else np.nan


    days = (eq.index[-1] - eq.index[0]).days
    CAGR = (1 + total_pnl / capital) ** (365 / total_days) - 1 if total_days > 0 else np.nan

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


# In[115]:


metrics = boros_metrics(trades_df3, equity_df3)
metrics


# In[68]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

#plt.plot(merged.index, merged["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"], label="1", linewidth=2)
#plt.plot(merged.index, merged["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"], label="2", linewidth=2)
#plt.plot(merged.index, merged["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"], label="3", linewidth=2)
#plt.plot(merged.index, merged["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"], label="4", linewidth=2)
#plt.plot(merged.index, merged["gap_zscore_2"], label="gap_zscore_2", linewidth=2)



plt.plot(merged.index, merged["fundingAPR"], label="Funding", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Comparison_APR — Implied")
plt.xlabel("Date")
plt.ylabel("APR (%)")
plt.tight_layout()
plt.show()


# In[76]:


merged['gap<>2']=merged['0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f']-merged['fundingAPR']
T = 10   # days horizon proxy
S= 30

merged["gap_MA_2"] = merged['gap<>2'].ewm(span=T, min_periods=T).mean()
merged["gap_std_2"] = merged['gap<>2'].rolling(S, min_periods=S).std()
merged['gap_zscore_2']=(merged['gap<>2']-merged["gap_MA_2"])/merged["gap_std_2"]
#merged.tail()


# In[116]:


def backtest_yu_gap_pro_2(
        df,
        capital=10000,
        entry_z=1.0,
        exit_z=0,
        alloc_pct=0.25
    ):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df)-1):

        today = df.iloc[i]
        tomorrow = df.iloc[i+1]

        ts_today = df.index[i]
        ts_t1    = df.index[i+1]

        z_today = today["gap_zscore_2"]
        ma3     = today["realized_ma3"]

        # ========== ENTRY SNAPSHOT ==========
        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (
            (realized_entry > implied_entry*1.5) 
        )

        # ========== ENTRY ==========
        if open_trade is None and not np.isnan(z_today):

            notional = free_equity * alloc_pct

            if z_today < -entry_z and not block_long:
                open_trade = {
                    "side": "LONG_YU",
                    "position": 1,
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
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
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

        # ========== ACTIVE TRADE ==========
        if open_trade and i >= open_trade["entry_idx"]:

            realized_today = today["fundingAPR"]

            if open_trade["position"] == 1:
                pnl = open_trade["notional"] * (realized_today - open_trade["entry_implied"]) * dt
            else:
                pnl = open_trade["notional"] * (open_trade["entry_implied"] - realized_today) * dt

            open_trade["trade_pnl"] += pnl
            equity += pnl
            free_equity += pnl    # funding loss reduces free collateral

            # ---------- STOPLOSS (Funding shock) ----------
            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2

            # ---------- EXIT ----------
            days_held = (ts_today - open_trade["entry_date"]).days

            if abs(z_today) < exit_z or shock or days_held >= 90:


                trades.append({
                    "side": open_trade["side"],
                    "entry_date": open_trade["entry_date"],
                    "exit_date": ts_today,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": open_trade["entry_implied"],
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": today["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"],
                    "exit_realized": realized_today,
                    "days_held": (ts_today - open_trade["entry_date"]).days,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%":(pnl / (capital * days_held / 365)) * 100 if days_held > 0 else np.nan

                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades),df


# In[117]:


trades_df2,equity_df2 = backtest_yu_gap_pro_2(
        merged,
        capital=10000,
        entry_z=1.0,
        exit_z=0.01,
        alloc_pct=0.5
    )
trades_df2


# In[118]:


trades_df2['ROI_%'].sum()


# In[119]:


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

    ROI = (total_pnl / (capital * total_days / 365)) * 100 if total_days > 0 else np.nan


    days = (eq.index[-1] - eq.index[0]).days
    CAGR = (1 + total_pnl / capital) ** (365 / total_days) - 1 if total_days > 0 else np.nan

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


# In[120]:


metrics = boros_metrics(trades_df2, equity_df2)
metrics


# In[64]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(merged.index, merged["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"], label="1", linewidth=2)
#plt.plot(merged.index, merged["0x48e1e85ab2d717a41cb7aeaf0de12321b8c14a0f"], label="2", linewidth=2)
#plt.plot(merged.index, merged["0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"], label="3", linewidth=2)
#plt.plot(merged.index, merged["0xed3f08b113b1d99e0885b82fcb24b544cd3a84a8"], label="4", linewidth=2)
#plt.plot(merged.index, merged["gap_zscore_1"], label="gap_zscore_1", linewidth=2)



plt.plot(merged.index, merged["fundingAPR"], label="Funding", linewidth=2)

plt.grid(alpha=0.3)
plt.legend()
plt.title("Comparison_APR — Implied")
plt.xlabel("Date")
plt.ylabel("APR (%)")
plt.tight_layout()
plt.show()


# In[124]:


merged['gap<>1']=merged['0x0ee61be6f8f73fb1412cc1205fb706ea441901ab']-merged['fundingAPR']
T = 10   # days horizon proxy
S= 30

merged["gap_MA_1"] = merged['gap<>1'].ewm(span=T, min_periods=T).mean()
merged["gap_std_1"] = merged['gap<>1'].rolling(S, min_periods=S).std()
merged['gap_zscore_1']=(merged['gap<>1']-merged["gap_MA_1"])/merged["gap_std_1"]
#merged.tail()


# In[125]:


def backtest_yu_gap_pro_1(
        df,
        capital=10000,
        entry_z=1.0,
        exit_z=0,
        alloc_pct=0.25
    ):

    import numpy as np
    import pandas as pd

    df = df.copy()
    dt = 1 / 365

    df["realized_ma3"] = df["fundingAPR"].rolling(3).mean()

    equity = capital
    free_equity = capital
    df["equity"] = np.nan

    trades = []
    open_trade = None

    for i in range(len(df)-1):

        today = df.iloc[i]
        tomorrow = df.iloc[i+1]

        ts_today = df.index[i]
        ts_t1    = df.index[i+1]

        z_today = today["gap_zscore_1"]
        ma3     = today["realized_ma3"]

        # ========== ENTRY SNAPSHOT ==========
        realized_entry = tomorrow["fundingAPR"]
        implied_entry  = tomorrow["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"]

        block_long = (
            (realized_entry < 0) or
            ((realized_entry < ma3) and (realized_entry < implied_entry * 0.5))
        )

        block_short = (
            (realized_entry > implied_entry*1.5) 
        )

        # ========== ENTRY ==========
        if open_trade is None and not np.isnan(z_today):

            notional = free_equity * alloc_pct

            if z_today < -entry_z and not block_long:
                open_trade = {
                    "side": "LONG_YU",
                    "position": 1,
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
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
                    "entry_date": ts_t1,
                    "entry_idx": i+1,
                    "entry_z": z_today,
                    "entry_implied": implied_entry,
                    "entry_realized": realized_entry,
                    "notional": notional,
                    "margin": notional,
                    "trade_pnl": 0
                }
                free_equity -= notional

        # ========== ACTIVE TRADE ==========
        if open_trade and i >= open_trade["entry_idx"]:

            realized_today = today["fundingAPR"]

            if open_trade["position"] == 1:
                pnl = open_trade["notional"] * (realized_today - open_trade["entry_implied"]) * dt
            else:
                pnl = open_trade["notional"] * (open_trade["entry_implied"] - realized_today) * dt

            open_trade["trade_pnl"] += pnl
            equity += pnl
            free_equity += pnl    # funding loss reduces free collateral

            # ---------- STOPLOSS (Funding shock) ----------
            shock = abs(realized_today) > abs(open_trade["entry_realized"]) * 2

            # ---------- EXIT ----------
            days_held = (ts_today - open_trade["entry_date"]).days

            if abs(z_today) < exit_z or shock or days_held >= 90:


                trades.append({
                    "side": open_trade["side"],
                    "entry_date": open_trade["entry_date"],
                    "exit_date": ts_today,
                    "entry_z": open_trade["entry_z"],
                    "exit_z": z_today,
                    "entry_implied": open_trade["entry_implied"],
                    "entry_realized": open_trade["entry_realized"],
                    "exit_implied": today["0x0ee61be6f8f73fb1412cc1205fb706ea441901ab"],
                    "exit_realized": realized_today,
                    "days_held": (ts_today - open_trade["entry_date"]).days,
                    "trade_pnl": open_trade["trade_pnl"],
                    "ROI_%":(pnl / (capital * days_held / 365)) * 100 if days_held > 0 else np.nan

                })

                free_equity += open_trade["margin"]
                open_trade = None

        df.iloc[i, df.columns.get_loc("equity")] = equity

    return pd.DataFrame(trades),df


# In[126]:


trades_df1, equity_df1 = backtest_yu_gap_pro_1(
        merged,
        capital=10000,
        entry_z=1.0,
        exit_z=0.01,
        alloc_pct=0.5
    )
trades_df1


# In[127]:


trades_df1['ROI_%'].sum()


# In[128]:


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

    ROI = (total_pnl / (capital * total_days / 365)) * 100 if total_days > 0 else np.nan


    days = (eq.index[-1] - eq.index[0]).days
    CAGR = (1 + total_pnl / capital) ** (365 / total_days) - 1 if total_days > 0 else np.nan

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


# In[129]:


metrics = boros_metrics(trades_df1, equity_df1)
metrics


# In[113]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

implied_col = "0x6e51252c47e6f3f246a6f501ac059d7d24a5a286"

implied = merged[implied_col].copy()
funding = merged["fundingAPR"].copy()
gap = merged["gap_zscore_3"].copy()

# detect flat-line expiry region
diff = implied.diff().abs()
flat_mask = diff < 1e-6
flat_run = flat_mask.rolling(7).sum()
cut_idx = flat_run[flat_run >= 7].index.min()

if pd.notna(cut_idx):
    implied = implied.loc[:cut_idx]
    funding = funding.loc[:cut_idx]
    gap = gap.loc[:cut_idx]

plt.figure(figsize=(12,6))
#plt.plot(implied.index, implied, label="Implied ", linewidth=2)
#plt.plot(funding.index, funding, label="Funding", linewidth=2)
plt.plot(gap.index, gap, label="Z_Score_Gap", linewidth=2)


plt.grid(alpha=0.3)
plt.legend()
plt.title("Z_Score_Gap ")
plt.xlabel("Date")
plt.ylabel("Zscore")
plt.tight_layout()
plt.show()


# In[80]:


merged.head()


# In[ ]:




