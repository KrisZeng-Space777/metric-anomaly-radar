# Method notes

This document records the decisions behind the code, including the ones that make
the numbers in the README trustworthy.

## 1. Why synthetic data

A detector can only be scored where the truth is known. Real ad logs never carry
a label saying "tracking broke on 12 May". Public anomaly benchmarks (NAB, Yahoo
S5, NASA telemetry) are also not shaped like advertising data: they have no
keyword packs, no hour-of-day mix and no cost-per-acquisition ratio.

So the repo simulates the delivery process and injects anomalies with a recorded
answer key. The trade-off is explicit: **results say how the detectors behave on
this generator, not on any real account**. The generator is deliberately
documented so a reader can judge how far the conclusions travel.

## 2. Data generating process

`synth.generate` builds the frame at `(date, keyword_pack, placement, hour_block)`:

| component | modelling choice |
|---|---|
| daily volume | base clicks × weekday factor × annual sine × payday bump × log-normal noise |
| pack split | fixed click shares, small per-pack jitter |
| price | per-pack CPC × placement multiplier × small noise |
| conversion | per-pack CVR, better on higher-intent placements, small drift |
| budget cap | per pack-day cap at 1.3× median, binding occasionally |

The cap matters: it makes the series slightly heavy-tailed, which is exactly the
regime where a mean/std z-score starts to fail.

## 3. Anomaly families

| family | shape | why it is realistic |
|---|---|---|
| `level_shift` | step ×1.55 for 8 days | budget rule changed, a pack paused |
| `spike` | one day ×2.6 | tracking outage / double counting |
| `trend_change` | ramp to ×1.5 over 14 days | bid landscape deteriorating |
| `variance` | mean-preserving log-normal shocks, σ×3 | unstable delivery, no mean move |

The variance family is included on purpose. It is the case where naive detectors
look good on paper and useless in practice, and reporting it keeps the benchmark
from being a vanity table.

## 4. Detector design

All three detectors share one constraint: **the decision at day *t* may only use
data up to day *t***.

- **`robust_z`** — subtract a trailing same-weekday median (kills the weekly cycle
  cheaply), then score the residual with median and MAD over a trailing window.
  MAD instead of standard deviation because spikes inflate σ until nothing looks
  anomalous (masking).
- **`stl_residual`** — refit STL on a trailing 56-day window at every step and
  score the last residual with the same robust scale. Fitting STL once on the
  whole series and reading residuals is the classic leakage bug: the seasonal
  component would already know about the anomaly.
- **`cusum`** — two-sided CUSUM on standardised first differences. Accumulates
  small deviations, so it reacts to steps and drifts that a point-wise score
  misses. The statistic resets after an alarm; without the reset, a single level
  shift alarms forever and the false-alarm count explodes.

## 5. Evaluation protocol

Two levels, always reported together:

- **Point-wise** precision / recall over every monitored series-day. Punishes
  detectors that fire on everything.
- **Event-wise** recall @ 3-day tolerance plus **MTTD** (mean time to detect).
  Punishes detectors that fire once at the start of an event and then go quiet.

An event counts as detected if any alarm falls in `[start − 1, start + tolerance]`.
The one-day lead allowance is deliberate: an alarm the day before an anomaly is
operationally useful, not an error. Alarms outside `[start − 1, end + 1]` are
counted as false alarms and normalised per 100 series-days, because raw counts
are not comparable across series of different lengths.

Anomalies are only injected **after a 45-day warm-up**, so no detector is judged
on a period where it had no history to learn from.

### Scope of an event

A site-wide tracking outage fires all ten monitored series on the same day. It is
one incident, not ten. Events therefore carry a `scope`:

- `scope="series"` — only the named series is affected; alarms on other series
  during the window are false alarms.
- `scope="all"` — every series is legitimately anomalous; the event is counted
  once, detectable in any series, and its window is masked from the false-alarm
  count everywhere.

Scoring a global event per series would mis-rank every detector in the same
direction, but the direction would still be wrong.

### Tuning protocol

Thresholds are not chosen on the data they are reported on. `scripts/tune_detectors.py`
sweeps window lengths and thresholds on a **development seed** (`seed=1`); the
selected configuration is then evaluated once on the **held-out seed** (`seed=7`)
and only those numbers appear in the README. The sweep is cheap to re-run and the
gap between the two seeds is a useful stability check.

### Known misses, explained

- **The 14-day ramp** moves +3.8%/day ≈ 0.34 robust σ/day, which is below CUSUM's
  0.5σ allowance — the accumulated statistic never crosses the threshold inside
  the 3-day tolerance. Lowering the allowance buys ramp sensitivity at the cost of
  a much noisier alarm stream; that trade was deliberately not taken here.
- **The variance event** keeps the mean and multiplies the noise. All three
  detectors threshold the deviation from a location estimate, so a mean-preserving
  second-moment change is invisible by construction. A rolling-volatility model
  would be the right tool and is the natural next detector.

## 6. Attribution

For additive metrics (spend, clicks, conversions) the change decomposes exactly
per member. For ratio metrics such as CPA = spend / conversions it does not: a
member can have a worse CPA and contribute nothing to the total change if it has
almost no volume.

`attribution.shapley_change` therefore treats members as players in a cooperative
game and computes Shapley values on the change:

```
v(S) = metric_current(S) − metric_baseline(S),    v(∅) = 0
```

So the contributions sum to `v(all)` — the total change — and interaction between
members is split evenly instead of being attributed to whoever is iterated last.
Exact enumeration is used up to 10 members (2^10 subset evaluations); above that
the function raises rather than silently approximating.

## 7. Limitations

- Single metric, univariate per series. Correlated anomalies across packs are not
  modelled, and a multivariate detector (e.g. on the full vector of series) would
  plausibly do better on the `total` series.
- Thresholds are fixed per run, not calibrated per series against an alarm budget.
- The variance family is only partially detectable by design; the reported recall
  is a lower bound on what a richer model (e.g. rolling volatility models) could do.
- The generator's seasonality is weekly plus annual. Detectors that assume
  stronger structure would need re-validation elsewhere.
- The STL comparison reflects *this generator* (short history, weekday-
  heteroskedastic noise). On series with longer history or richer seasonality the
  ranking can — and should — be re-measured with the same protocol.
- No real data is used anywhere in this repository.
