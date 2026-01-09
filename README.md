Detecting Crowd-Chasing in Funding Yield Markets on Boros: 
Systematic Trading of Mispricings Between Implied and Realized Funding Rates

Overview: 
This repository implements a crowding-aware forward-yield trading strategy for Boros Yield Units (YU).
Unlike traditional funding strategies that harvest carry, this framework monetizes expectation failures between:
Implied Funding APR embedded in Boros YU and Realized Funding APR observed in perpetual futures markets
The strategy detects crowd-chasing and panic regimes by normalizing the gap between implied and realized funding, generating event-driven long/short signals with convex payoff characteristics.

Core Idea:
Gap(t)​=f^​t​(T)−ft
where:

f^​t​(T) : Implied APR(from Boros YU)
ft : Realized funding APR

Forward yield markets adjust expectations with inertia.
During leverage expansions and liquidation cascades, funding moves faster than implied APR, creating repeatable mispricings.

Signal Construction:
Expectation Gap
gap = implied_apr - funding_apr

Z-Score Normalization:
T = 10 or 15   # EMA window
S = 30 or 45  # Rolling std window

gap_mean = gap.ewm(span=T, min_periods=T).mean()
gap_std  = gap.rolling(S, min_periods=S).std()

zscore = (gap - gap_mean) / gap_std

Trading Logic:

| Z-Score State | Market Condition                                       | Action       |
| ------------- | ------------------------------------------------------ | ------------ |
| Z > +1.0      | Implied too high vs funding (panic / funding collapse) | **Short YU** |
| Z < −1.0      | Funding too high vs implied (crowd-chasing)            | **Long YU**  |
| |Z| < 0.01    | Expectation normalization                              | **Exit**     |


Safety Filters:
- Long Entry Block
Block LONG YU if: Funding < 0 OR Funding < MA3(funding) and Funding < 0.5 × Implied
- Short Entry Block
Block SHORT YU if: Funding > 1.5 × Implied

- Funding Shock Stop-Loss
Force exit if: Funding APR > 2 * Funding APR(At entry)

- Max Holding Period
All trades are exited after 90 days.


PnL Attribution
For notional N and dt=1/365:
if position == 1:   # LONG YU
    pnl = N * (funding_today - entry_implied) * dt
else:               # SHORT YU
    pnl = N * (entry_implied - funding_today) * dt

PnL is accumulated daily and immediately impacts free collateral.


ROI Metric
Time-normalized capital efficiency: ROI% ​= (PnL/(Capital * (Days Held/365))) * 100


Synthetic Implied APR Simulator
Due to limited Boros history, implied APR is simulated as:

f^​t​(T) = EMA14​(ft​) + RPt + 0.6(T)^1/2 + ϵt​
Where:
- EMA14(ft) = expectation anchor
- RPt = 3-state Markov crowding regime
| State    | RP Value |
| -------- | -------- |
| risk_on  | +3.5     |
| neutral  | 0.0      |
| risk_off | −2.0     |

Transition Matrix:
​0.900  0.100  0.000
0.050  0.900  0.050
0.00​0  0.10​0  0.90
​​
- 0.6(T)^1/2 = maturity convexity
- ϵt ∼ N(0,0.5)


Stress Environment
Stress tests inject:
- Funding jumps (liquidation cascades)
- Fat-tailed noise
- Liquidity vacuum events
This evaluates robustness under regime-breaking market conditions.


Key Results
- Convex payoff distribution
- Bounded drawdowns
- Signal improves under stress
- Captures expectation normalization, not funding carry


Disclaimer
This repository is for research purposes only.
Boros markets are early-stage and subject to protocol, liquidity, and execution risks.
This framework is not financial advice.

