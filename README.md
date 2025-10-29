# DSA 210 — IB Dual-AVWAP (Alpha-Durable and Regime-Agnostic)

> Goal in one sentence:
> Using only 1-minute price bars (High, Low, Close) and 1-minute traded volume for SPY, we build three anchored VWAP lines from the first hour of the day (IB: 09:30–10:30 U.S. Eastern Time) and then predict the next 15 or 30 minutes (up or down) for the rest of the session (10:45–15:10 ET).
>
> Scope limits:
> This project does not place trades, it does not use order-book, latency, or fees.

---

## Table of Contents
1. Motivation  
1.5. Glossary
2. Data Sources  
3. Hypotheses
4. Methodology
5. Visualization Findings  
6. Hypothesis Testing
7. ML Model Implementation 
8. Limitations and Future Work

---

## 1. Motivation

Simple idea:  
During the first trading hour (IB = Initial Balance, 09:30–10:30 ET), we find the strongest 5-minute up move, the strongest 5-minute down move, and use the opening moment at 09:30:00. We start a separate Anchored-VWAP from each of these three moments:
- AVWAP up (anchored at the strongest 5-minute up burst)
- AVWAP down (anchored at the strongest 5-minute down burst)
- AVWAP open (anchored at the opening tick)

From 10:45 to 15:10 ET, for every minute, we predict whether price will be higher or lower after 15 minutes (Dir-15) and after 30 minutes (Dir-30).

Why this might help:
- The VWAP family (Volume-Weighted Average Price) is widely watched. Price relative to VWAP or AVWAP is often used as a simple fairness or stretch signal.
- Where price sits relative to our three AVWAPs (above, between, below) and whether crossings occur can carry short-term direction hints.
- A volume-clock percentage (how much of today’s total volume has already traded) can describe intraday behavior better than wall-clock time.

Fixed daily prediction window:
- Predict only in 10:45–15:10 ET, to let the first hour settle and to avoid very late closing flows.
- Data used is strictly 1-minute OHLC and 1-minute Volume for SPY.

---

## 1.5. Glossary

- SPY: A U.S. exchange-traded fund that tracks the S&P 500 index. We use SPY because it is liquid and has clean 1-minute data.
- Bar or candle (1-minute): Summary of trading in one minute. It shows High (maximum price in that minute), Low (minimum price), Close (last price in the minute), and Volume (number of shares traded).
- Volume: How many shares changed hands. A high-volume minute means many trades happened.
- VWAP (Volume-Weighted Average Price): An average price where each trade is weighted by traded size. It is a fairness-style average that pays more attention to busy moments.
- Anchored VWAP (AVWAP): A VWAP that starts at a specific time point and then updates forward. For example, AVWAP anchored at 09:30 uses all data from 09:30 up to the current minute.
- IB (Initial Balance): The first hour after the market opens, 09:30–10:30 ET.
- Strongest 5-minute up or down move: Among all overlapping 5-minute windows inside 09:30–10:30, the window with the largest rise (up) or largest drop (down) by simple measures.
- AVWAP up, AVWAP down, AVWAP open: Three AVWAP lines anchored at the strongest up window start, the strongest down window start, and the opening tick at 09:30.
- State: Where the current price sits relative to a pair of AVWAP lines (above both, between them, below both). We compute this for pairs and can extend to include AVWAP open when needed.
- Cross event: A crossing by price over an AVWAP, or a crossing between two AVWAP lines.
- Delta VWAP (ΔVWAP): Distance between two AVWAP lines. Large means far apart; small means close together.
- Volume-clock percentage: Share of the day’s total volume that has already traded. For example, 35 percent means 35 percent of the day’s expected total shares have already traded.
- Dir-15 and Dir-30: Labels we want to predict. For minute t, Dir-15 is up if Close at t+15 minutes is greater than Close at t; otherwise down. Dir-30 uses a 30-minute horizon.
- Accuracy, precision, recall, F1: Standard classification metrics. Tooltips in notebooks explain their meaning.
- Cliff’s delta: An effect-size measure showing how separated two distributions are. Larger absolute values indicate a stronger effect.
- Walk-forward validation with purging: Train on past blocks and test on later blocks, while removing overlaps that could leak future information.

---

## 2. Data Sources

- Instrument: SPY at 1-minute resolution
- Content: OHLC (High, Low, Close) and Volume only
- Date range: about 40 to 60 trading days as a start
- Timezone: U.S. Eastern Time (ET)
- Cleaning: flag or remove zero or obviously wrong volume bars; keep raw and clean CSVs cached
- Folders:
  - data/raw for raw downloads
  - data/clean for cleaned files
  - data/cache for intermediate outputs such as features, labels, and splits

Pipeline should be deterministic and reproducible.

---

## 3. Hypotheses

We will measure hit-rate versus 50 percent, one-sided p value less than 0.05, and effect size using Cliff’s delta.

- H1 reversion in between state: When price is between two AVWAPs, delta VWAP is small, and their slopes point in opposite directions, the next 15 to 30 minutes tend to revert toward the middle more often than 50 percent.
- H2 continuation when both agree: When price is above both AVWAPs or below both and the two AVWAP slopes agree, continuation over 30 minutes is above 50 percent.
- H3 third anchor and volume-clock helps: Adding AVWAP open and using volume-clock percentage yields better balance of hits and errors than using only two anchors.
- H4 volume-clock sensitivity: Signal strength differs across volume-clock bands.
- H5 short-horizon alignment after a cross: Just after a cross event, the average direction over the next 15 to 30 minutes tends to match the crossing direction, except for very short whipsaws.
- H6 AVWAP ordering matters: Specific orderings of AVWAP open, up, and down correlate with more consistent future direction.
- H7 IB width effect: On narrow IB days, reversion is more common; on wide IB days, continuation is more common.
- H8 gap alignment: If the opening gap direction and the AVWAP slopes agree, the 30-minute hit-rate improves.
- H9 middle delta VWAP is a sweet spot: Signals are more stable when delta VWAP is moderate. When it is very large, avoid extreme bets on either pure reversion or pure continuation.

Success thresholds per hypothesis:
- Hit-rate at least 55 percent in the filtered subcase and one-sided p less than 0.05
- Effect size absolute Cliff’s delta around 0.147 or larger
- Classifier metrics including accuracy, precision, recall, and F1

---

## 4. Methodology

1) Fix the prediction window  
Generate labels only in 10:45–15:10 ET.

2) Find strongest 5-minute moves inside IB  
Slide a 5-minute window across 09:30–10:30 and compute:
- Body move from close to close
- True-range move from high to low  
Pick the start minute of the window with the strongest up move and the strongest down move.

3) Build three AVWAP lines forward  
AVWAP up, AVWAP down, and AVWAP open.  
At every minute t, each AVWAP uses all minute bars from its anchor up to t. No future data is used.

4) Create features from OHLCV only  
Examples:
- State relative to AVWAP pairs
- Delta VWAP between AVWAP pairs
- AVWAP slopes using short windows
- Cross flags for price crossing AVWAP and AVWAP crossing AVWAP
- IB range percentage relative to today’s level
- Opening gap percentage versus previous close
- Short volatility using 5-minute and 15-minute windows
- Trend score from recent high, low, close
- Volume-clock percentage

5) Create labels without leakage  
For each minute t in 10:45–15:10:
- Dir-15 equals 1 if Close at t+15 is greater than Close at t, else 0
- Dir-30 defined similarly over 30 minutes  
All features use only data at or before t. Labels use t to t plus horizon.

6) Validate fairly with walk-forward and purging  
Split by weeks or days. Train on earlier blocks, test on later blocks.  
Purge overlaps to avoid accidental look-ahead.  
Report accuracy, precision, recall, F1, Cliff’s delta, and bootstrap by day confidence intervals.

---

## 5. Visualization Findings

- Heatmaps of State by Delta VWAP versus Dir-15 and Dir-30 hit-rates
- Cross event paths showing average path before and after price or AVWAP crossings
- Volume-clock maps showing where signals strengthen or weaken
- AVWAP ordering panels conditioned on AVWAP open, up, down
- IB width slices that compare narrow, medium, and wide IB days

---

## 6. Hypothesis Testing 

For each hypothesis:
1. Filter the data to the relevant condition
2. Compare hit-rate to 50 percent using a one-sided test with p less than 0.05
3. Compute Cliff’s delta with a bootstrap confidence interval
4. Report accuracy, precision, recall, and F1 on the filtered subset
5. Run sensitivity checks by volume-clock percentage, IB range percentage, delta VWAP bins, and AVWAP ordering
6. Decide accepted, rejected, or limited support

---

## 7. ML Model Implementation

Philosophy: keep it small and interpretable. The system outputs up or down only.

### 7.1 Router to detect the day’s mode
- Purpose: choose between range-like behavior and trend-like behavior
- Inputs: IB range percentage, opening gap percentage, short volatility over 5 and 15 minutes, trend score from H L C, volume-clock percentage, absolute delta VWAP, and state
- Model: a small rule-based tree with two or three levels or a small linear classifier with a single threshold
- Output: range-like or trend-like

### 7.2 Prediction labeler
- Range-like: prefer reversion rules for Dir-15 such as state equals between, small delta VWAP, opposite AVWAP slopes, and no fresh breakouts
- Trend-like: prefer continuation rules for Dir-30 such as above-both or below-both, same slopes, moderate delta VWAP, and no recent whipsaw crosses
- Model size is capped, for example maximum depth three if using trees

### 7.3 Evaluation
- Use walk-forward with purging
- Report accuracy, precision, recall, F1
- Report Cliff’s delta and bootstrap by day uncertainty bands

---

## 8. Limitations and Future Work

- Noisy 1-minute data: a few bars can be erroneous or zero-volume. We flag or remove them.
- No outside information: we do not use news, fear indexes, or order-book data, so sudden jumps may be unexplained.
- Overfitting risk: we keep models small and use honest walk-forward splits, but risk cannot be fully removed.
- Transfer to other tickers: thresholds likely need retuning for symbols other than SPY.
- Next steps: extend date range, include multiple regimes, add more symbols, stay with OHLCV only, and explore small improvements around the fixed window.

---
