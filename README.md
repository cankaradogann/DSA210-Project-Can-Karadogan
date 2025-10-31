# Day Trading Volume Weighted Average Price Strategy Project

> Goal in one sentence:
>
> We study the U.S. stock market using SPY, which is an ETF that follows the S&P 500, and we use 1-minute price and volume data from the first hour of trading (09:30–10:30 New York time) to draw three AVWAP (anchored volume-weighted average price) lines; these lines show price levels where trading was important, and then test whether these volume-weighted levels help us tell future stock price actions intraday for traders.

---

## Table of Contents

1. [Motivation](#1-motivation)  
1.5. [Glossary](#15-glossary)  
2. [Data Sources](#2-data-sources)  
3. [Hypotheses](#3-hypotheses)  
4. [Methodology](#4-methodology)  
5. [Visualization Findings](#5-visualization-findings)  
6. [Hypothesis Testing](#6-hypothesis-testing)  
7. [ML Model Implementation](#7-ml-model-implementation)  
8. [Limitations and Future Work](#8-limitations-and-future-work)

---

## 1. Motivation

**Simple description**  
On a normal U.S. stock market day, the market opens at 09:30 in New York time. We care most about the first 1 hour after the market opens, from 09:30 to 10:30. Traders often call this first 1 hour the “initial balance.” It is important because most of the day’s big buyers and sellers show their first actions in this period.

Inside this first 1 hour, we look at SPY. SPY is an ETF that follows the S&P 500 index, so you can think of it as “the price of the U.S. stock market.” In this first hour, we try to find two very strong 5-minute moves:

- the 5-minute part where price went up the most (the strongest up move),
- the 5-minute part where price went down the most (the strongest down move).
- We also mark the exact opening time, 09:30:00.

So we have 3 important starting points:

- the strongest up move,
- the strongest down move,
- the exact market open at 09:30.

After we create these 3 AVWAP lines, we do not use the whole trading day; instead we focus only on the cleaner part of the day, from 10:45 to 15:30 New York time. We start at 10:45 because the 10:30–10:45 period is often noisy and unstable, so it can give us bad or random signals. We stop at 15:30 because the last 30 minutes (15:30–16:00) is usually very volatile: many big traders close their positions and this makes price move in a way that is not normal for the rest of the day.

So we choose 10:45–15:30 to get more “normal” market behavior. In this time window we check whether our 3 AVWAP lines are actually useful: if the AVWAP lines really show direction and if they can help us predict the next price move.

**Why this could be useful**  
This method can be useful because VWAP and anchored VWAP are price lines that give more importance to the minutes when a lot of trading happens, and less importance to the minutes when almost nobody trades. Many traders and funds look at these lines to see a kind of “fair price” for the day. If the current price is above these lines, it can mean buyers are stronger right now; if it is below, it can mean sellers are stronger.

Also, how these 3 lines stand compared to each other (for example all three under the price, or price stuck between them) gives a quick picture of whether the market is in balance or out of balance. We skip the first hour and the last minutes of the day because those times are too noisy and too emotional; by working only in the calmer middle part of the day, we can test our rules in a cleaner, more realistic way.

**Key design choices to keep the build easy**  
We work only with 1-minute open-high-low-close and volume. We keep features small and clear. We validate using walk-forward splits to mimic real time learning without peeking into the future.

---

## 1.5. Glossary

**SPY**  
SPY is an exchange-traded fund that tries to track the S&P 500 index. An exchange-traded fund is a basket of many stocks that is traded like a single stock. We use SPY because it trades a lot every day, the data is widely available, and one-minute data is usually clean. Using SPY helps reduce special events that single companies can have.

**Bar or candle**  
A bar is a summary of trading during a fixed time window. A one-minute bar includes the highest price seen in that minute, the lowest price seen in that minute, the last price of that minute, and the number of shares traded in that minute. These four numbers are called high, low, close, and volume. We use one-minute bars because they keep enough detail for short-term analysis while staying simple to work with.

**Volume**  
Volume is the count of shares that changed hands in a given minute. If volume is high, many shares traded. If volume is low, few shares traded. Volume helps weight the VWAP so that busy minutes count more than quiet minutes.

**VWAP**  
VWAP means volume-weighted average price. It is an average that gives higher weight to minutes with more volume. The purpose of VWAP is to describe a fair or typical price where most trading happened. If the price is far above VWAP, it may be considered stretched up. If the price is far below VWAP, it may be considered stretched down.

**Anchored VWAP or AVWAP**  
Anchored VWAP is a VWAP that starts at a chosen moment and moves forward in time as each new minute arrives. The anchor can be the open, a special event, or any chosen time. The anchored VWAP from time A to the current minute uses only bars from time A up to now. The purpose of AVWAP is to freeze the start time so we know exactly what price history is being summarized.

**Initial balance or IB**  
The initial balance is the first hour of the regular trading session. In U.S. Eastern Time this is 09:30–10:30. Many traders consider this first hour special because it often sets the day’s early high and low and defines the early character of the day. We use the initial balance to locate strong bursts that may shape intraday direction.

**Strongest five-minute up or down move**  
This is the five-minute window inside the initial balance that shows the largest simple move. We can measure the move in two ways. One way is close-to-close change from the first minute to the fifth minute. Another way is the high-low range within the same five minutes. We pick the window that gives the largest increase for the up move and the largest decrease for the down move. The purpose is to capture the moments with the most decisive push.

**AVWAP up, AVWAP down, AVWAP open**  
These are the three anchored VWAP lines we use. AVWAP up is anchored at the start time of the strongest up window. AVWAP down is anchored at the start time of the strongest down window. AVWAP open is anchored exactly at 09:30:00. The purpose of having three anchors is to describe three different stories of the day. The up anchor captures early aggressive buying. The down anchor captures early aggressive selling. The open anchor captures the general crowd entry point.

**State**  
State describes where the current price sits with respect to a pair of AVWAP lines. The three basic states are above both lines, between the two lines, or below both lines. The purpose of the state is to compress location information into a few categories that are easy to visualize and test.

**Cross event**  
A cross event is when the price crosses an AVWAP line or when one AVWAP line crosses another AVWAP line. A cross can mark a possible change in control between buyers and sellers or a shift in the balance point defined by anchored averages. We test whether short-term direction after a cross tends to line up with the cross direction.

**Delta VWAP**  
Delta VWAP is the absolute distance between two AVWAP lines at the same minute. A large distance means the two anchored summaries disagree strongly. A small distance means the two anchored summaries are close. This distance can indicate whether the market is stretched or balanced between two anchor stories.

**Opening gap**  
The opening gap is the percentage difference between today’s opening price and yesterday’s closing price. A positive gap means today started higher than yesterday ended. A negative gap means today started lower. The purpose of the gap is to capture overnight pressure that may set the tone for the morning.

**IB width**  
IB width is the high minus the low inside the initial balance, scaled by the opening price to form a percentage. A narrow width suggests a quiet first hour. A wide width suggests a very active first hour. This can guide whether reversion or continuation is more likely.

**Cliff’s delta**  
Cliff’s delta is an effect size that compares two distributions by the probability that a random sample from one distribution is greater than a random sample from the other. Values near zero mean little separation. Large positive or negative values mean strong separation. We use it to describe practical impact, not just statistical significance.

**Walk-forward validation and purging**  
Walk-forward means we train on earlier blocks of days and test on later blocks. This avoids learning from the future. Purging means we remove samples near the training and testing boundary that could leak information due to overlapping label windows. These steps protect against overly optimistic results.

---

## 2. Data Sources

**Instrument and resolution**  
We use SPY ETF's one-minute bars and volume data. This resolution is a practical compromise between detail and computational cost. It is high enough to see short-term patterns but low enough to keep code and storage simple.

**Fields used**  
We use only high, low, and close prices, and volume for each minute. We do not use the open for each minute because the close is a standard reference for one-minute bars and makes the label definition simple.

**Date range**  
Start with about previous forty to sixty trading days. This is small enough to build and debug quickly. It is large enough to capture different day types. Later we can extend to more months.

**Timezone**  
All times are in U.S. Eastern Time to match the exchange session and the usual definition of the initial balance.

**Folder structure**  
We store raw downloads in data/raw. We store cleaned files in data/clean. We store computed outputs like features, labels, and splits in data/cache. This separation makes it easier to debug and to rerun steps without mixing files.

**Reproducibility**  
We aim for a deterministic pipeline. Given the same inputs and code version, the outputs should be the same. This helps comparison across experiments.

---

## 3. Hypotheses

**H1: Reversion in between state**  
When the price is between two AVWAP lines, the delta VWAP is small, and the two AVWAP slopes point in opposite directions, the next 15 to 30 minutes tend to revert toward the middle more often than 50 percent. The idea is that when two anchors disagree and are close, the price may be pulled to the midpoint.

**H2: Continuation when both agree**  
When the price is above both AVWAP lines or below both, and the two AVWAP slopes point in the same direction, continuation over 30 minutes is above 50 percent. The idea is that two agreeing anchors may reflect a stronger and more stable drift.

**H3: Third anchor improves stability**  
Adding the AVWAP open to the pair of up and down anchors improves the balance of hits and errors versus using only two anchors. The idea is that the open anchor captures the general starting point of the crowd and stabilizes decisions.

**H4: Cross alignment**  
Just after a cross event, the average direction over the next 15 to 30 minutes tends to match the cross direction, except for very short whipsaws. The idea is that crossing a fair value line can mark a shift in control.

**H5: AVWAP ordering**  
Specific orderings of AVWAP open, AVWAP up, and AVWAP down correlate with more consistent future direction. The idea is that the relative stack of anchors encodes a day story.

**H6: IB width effect**  
On narrow initial balance days, reversion is more common. On wide initial balance days, continuation is more common. The idea is that the amount of early expansion shapes what comes next.

**H7: Gap alignment**  
If the opening gap direction and the AVWAP slopes agree, the 30-minute hit-rate improves. The idea is that overnight pressure and anchored drift line up to push in the same direction.

---

## 4. Methodology

**Step 1: Define the prediction window**  
We only create labels from 10:45 to 15:10 ET. This choice avoids the messy open and the noisy close. It also creates a fixed daily window that is easy to manage.

**Step 2: Find strongest five-minute moves in the initial balance**  
We slide a five-minute window starting at each minute between 09:30 and 10:25. For each window we compute two measures. The first is the close-to-close change from minute one to minute five. The second is the high-low range within the window. We identify the start minute of the largest up move and the start minute of the largest down move. If the two definitions disagree, we prefer close-to-close for simplicity, and we record the other as a check.

**Step 3: Build three anchored VWAP lines**  
We build AVWAP up from the start of the strongest up window. We build AVWAP down from the start of the strongest down window. We build AVWAP open from 09:30:00. At each minute t, each AVWAP uses only the bars from its anchor up to t. This prevents using future information.

**Step 4: Compute simple features from OHLCV only**  
We keep features small and explainable.

*a. State features*  
We compute the location of price relative to a pair of AVWAP lines as above both, between, or below both. We can do this for each pair: up versus down, open versus up, open versus down.

*b. Delta VWAP features*  
We compute the absolute difference between the AVWAP values of a pair at time t. A small difference suggests balance. A large difference suggests strong separation.

*c. Slopes of AVWAP lines*  
We approximate the slope by taking the change in AVWAP over the past k minutes divided by k. We use k equal to 5 or 10 minutes. The slope direction helps detect rising or falling anchored averages.

*d. Cross flags*  
We set flags if the price crossed an AVWAP within the last k minutes or if two AVWAP lines crossed in the last k minutes. These binary markers identify recent structural changes.

*e. Opening gap and IB width*  
We compute the opening gap as the percentage difference between today’s open and yesterday’s close. We compute IB width as the high minus the low during 09:30–10:30 divided by the open. These help classify the day’s character.

*f. Short volatility and trend score*  
We compute short volatility as rolling high-low ranges over 5 and 15 minutes. We compute a simple trend score as the location of the close within a rolling high-low channel over the last m minutes. Values near the top of the channel suggest upward pressure. Values near the bottom suggest downward pressure.

**Step 5: Create leakage-free labels**  
For each minute t between 10:45 and 15:10, we set Dir-15 to 1 if the close at t plus 15 minutes is greater than the close at t, otherwise 0. We set Dir-30 in the same way but with 30 minutes. All features at time t must be computed using only information at or before time t. We avoid using any data from the future.

**Step 6: Validate with walk-forward and purging**  
We split the data into blocks by week or by sets of days. We train on earlier blocks and test on later blocks. We purge samples near the boundaries for at least the length of the longest label horizon to avoid hidden leakage. We report accuracy, precision, recall, F1, and Cliff’s delta on the test sets. We compute bootstrap by day confidence intervals for hit-rate and F1 to express uncertainty.

---

## 5. Visualization Findings

We plot the hit-rate for each state and for ranges of delta VWAP. This shows where reversion or continuation patterns are strongest.

**Cross event path studies**  
We align many days at the time of a cross event and average the price path before and after the cross. This shows the typical move around crosses.

**AVWAP ordering panels**  
We group minutes by the ordering of AVWAP open, AVWAP up, and AVWAP down. We then show hit-rates within each group. This tests whether the stack order relates to direction.

**IB width slices**  
We split days into narrow, medium, and wide initial balance groups and compare hit-rates. This tests whether early expansion predicts later behavior.

---

## 6. Hypothesis Testing

**Filter**  
For each hypothesis, we create a filter that selects the relevant minutes. For example, for reversion in the between state, we select minutes where the price is between two AVWAP lines, the delta VWAP is small by a fixed threshold, and the slopes point in opposite directions.

**Test**  
We compute the hit-rate on the filtered set. We run a one-sided test to check if the hit-rate is greater than 50 percent. We report the p value.

**Effect size**  
We compute Cliff’s delta for the distribution of signed returns over the horizon. This shows whether the effect is not only statistically significant but also practically meaningful.

**Classifiers**  
We also report accuracy, precision, recall, and F1 of simple rule-based or tree-based labelers restricted to the same filtered subset. This cross-checks that the effect carries into a model.

**Sensitivity checks**  
We repeat the analysis in bins of IB width, bins of delta VWAP, bins of opening gap sign, and different AVWAP orderings. The purpose is to check if the effect is robust across day types.

**Decision**  
We mark the hypothesis as accepted, rejected, or having limited support based on hit-rate, p value, effect size, and consistency across sensitivity checks.

---

## 7. ML Model Implementation

We keep the model simple and interpretable. The output is only up or down. Every rule should be readable as plain sentences that a non-technical reader can follow.

**Router to detect the day mode**  
*Purpose*  
The router chooses between range-like behavior and trend-like behavior. The router is not meant to be perfect. It is only a coarse switch to choose the right set of rules for the current conditions.

*Inputs*  
The router uses IB width, opening gap, short volatility over 5 and 15 minutes, a simple trend score from recent highs and lows, the magnitude of delta VWAP, and the state with respect to AVWAP pairs. Each input has a clear meaning and a bounded range.

*Model*  
We use a small decision tree with depth at most two or three, or we use a single linear rule with a threshold. A shallow model helps avoid overfitting and keeps decisions transparent.

*Output*  
The router returns range-like or trend-like. This output chooses which labeler to apply in the next step.

**Prediction labeler**  
*Range-like labeler*  
In range-like conditions, we prefer reversion rules for the 15-minute horizon. Typical signals include being between two AVWAP lines, having a small delta VWAP, seeing opposite AVWAP slopes, and not having a fresh breakout in the last few minutes.

*Trend-like labeler*  
In trend-like conditions, we prefer continuation rules for the 30-minute horizon. Typical signals include being above both or below both AVWAP lines, having the same AVWAP slope direction, having a moderate delta VWAP, and not seeing a recent whipsaw cross.

*Model size*  
We cap the tree depth at three. This limit forces clarity and reduces the risk of fitting noise.

*Evaluation*  
We use walk-forward with purging. We report accuracy, precision, recall, and F1 on test blocks. We also report Cliff’s delta so the reader can see variability.

---

## 8. Limitations and Future Work

**Noisy one-minute data**  
One-minute bars sometimes include errors or zero volume. We remove these cases, but small glitches can remain. This noise can create false crosses or short-lived spikes.

**No outside information**  
We do not use news, macro events, or order-book signals. Sudden jumps can occur without any warning in our features. This choice keeps the build simple but reduces context.

**Overfitting risk**  
Even with simple models and honest splits, time series can trick us. We reduce this risk by using few features, small trees, and strong validation. But the risk cannot be removed completely.

**Transfer to other tickers**  
If we apply the same rules to different symbols, thresholds may need to change. Liquidity, volatility, and day structure can differ across instruments.

**Next steps**  
We can extend the date range to include more regimes. We can add more symbols to test generality. We can explore mild extensions like time-of-day bins and gentle transforms of delta VWAP. We keep the volume-clock idea out to maintain simplicity.

---
