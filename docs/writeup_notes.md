# Write-up notes (for the README)

## Evaluation lessons
- No single method wins every year. Weaknesses show up exactly where the series changes:
  a structural break (2022, after the cancelled festivals) and a level shift
  (2026, falling Wikipedia traffic). Report errors per period, not one number.
- Don't judge a method by one series. Both snaive_364 (looked strong on Oktoberfest) and
  season_corr (looked best on Oktoberfest) told a different story across all 553 articles.
- Across many series, the headline metric is the per-article relative error
  (geometric mean of MAE ratios vs. a baseline). Pooled MASE is distorted by articles that
  were nearly dormant before becoming popular (near-zero scale). Keep MASE for
  single-series comparisons, where it works fine.
- Never tune on the test folds. All settings (Prophet event windows −14/+7 days,
  LightGBM parameters) were fixed up front. Tuning would need a separate validation period.

## Leakage
- Proven, not assumed: rebuilding features with all data after the origin deleted gives
  identical values (max difference 0.0). Repeated after adding season_corr.
- Second rule: training rows must be filtered by *target date* ≤ cutoff, not just origin date.

## Single series (Oktoberfest)
- prophet_events wins overall (MAE 251) using the festival calendar rule as known events.
  Caveat: hand-crafted domain knowledge. It works brilliantly for one article but doesn't
  scale to 553, where you can't write a rule per topic. That's the key tension:
  generic features that work everywhere vs. specific knowledge that works in one place.
- Remaining errors for prophet_events: the second-weekend peak is underestimated
  (~9,200 vs. 11,400) and the jump at the very end of the window is missed by all models.
  Untried idea: German public holidays (Prophet `add_country_holidays("DE")`).
- lgbm_single is second (292) and beats prophet_events in 2023–2025; it loses in 2022, when
  its seasonal hints point to the two cancelled festivals.

## Global model (553 articles)
- Halves the typical error vs. "same day last year" and is better on 99% of articles.
- Blind spot: strongly seasonal articles (Super Bowl, NFL, MLK Jr., Oktoberfest). Across
  most articles last year is a weak hint, so the model learned to distrust it everywhere.
- Two fixes compared:
  - article ID: slightly larger gain (~2% typical, 66% of articles; most seasonal articles
    improve), but it's memorisation and can't help new articles (cold start);
  - season_corr (seasonal reliability): generalises to any article with two years of
    history, but only helps where the yearly pattern is very regular (Oktoberfest, NFL).
  - Neither closes the gap on sharp annual spikes.
- My first conclusion ("the generalising feature beats the memorising one") came from
  Oktoberfest alone and turned out wrong across all articles.
- Sharper seasonal hints (exact day and 7-day max one year earlier) helped modestly
  (−0.8% typical, Oktoberfest −13%), but didn't unlock the spikes. The remaining limit is the
  model's conservatism in rare situations, not missing information. Next step would be
  loosening the objective/leaf size, tuned on a separate validation period.
- Production model: global + season_corr + article ID + sharp hints (best typical error);
  unknown articles fall back to the generic features.
- Production intervals: raw 72.5% → 80.2% after rolling conformal calibration
  (same picture as the earlier model).

## Intervals
- Raw quantile intervals were overconfident: 72.5% coverage for an 80% interval,
  worse at long horizons (79% → 70%). Miss direction followed drift (above in 2022,
  below in 2026).
- Rolling conformal calibration (per horizon, last 12 past folds) restored 80.2%, flat
  across horizons and years, at ~10% wider intervals.
- Calibration fixes the overall size of the uncertainty, not the model's blind spots: a
  single correction per horizon can't single out rare peaks (previous model: Oktoberfest
  festival only 71.8% → 75.5% on folds 7–55).
- With the sharp seasonal hints, Oktoberfest's festival coverage reached 79.7% raw /
  81.7% calibrated (folds 7–55), up from 71.8% / 75.5%: q90 now reaches ~14,000 on opening
  day while the median stays ~6,500. The upper-quantile model learned that a spike is
  possible, even though the median model stays conservative.

## Data limitations
- Article renames split pageview histories (e.g. William, Prince of Wales), so some series
  start late or jump. The 3-year history filter catches most.
- 44 articles excluded with documented reasons (configs/excluded.txt).
- One monthly top list (Sep 2024) was missing from the API; 35 of 36 months used.
- macOS setup needs `brew install libomp` for LightGBM (not a Python dependency).