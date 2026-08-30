# Evaluation and Comparison Protocol

## Primary Metrics in Return Space

- MAE
- RMSE
- R-squared
- Pearson information coefficient
- Spearman rank information coefficient
- Directional accuracy
- MSE skill versus zero-return baseline

```text
MSE skill = 1 - MSE_model / MSE_zero_return
```

Positive skill means improvement over predicting zero return. Negative skill means worse performance.

Directional accuracy excludes nearly zero true returns using the configured epsilon. A zero prediction counts as a miss. Do not describe the zero-return baseline as having 50% directional accuracy.

## Secondary Price Metrics

Reconstructed-price MAE and RMSE are reported only as secondary presentation metrics. Main conclusions use return-space metrics.

## Fair Internal Comparison

A valid model comparison requires:

- same raw data version,
- same target definition,
- same split boundaries,
- same feature set unless feature set is the controlled variable,
- same timestamps,
- same scaling policy,
- same metric implementation.

## 1h Versus 5m

The direct comparison uses only common hourly anchor timestamps and identical one-hour targets. Report sample counts after intersection.

## Final Uncertainty

E10 reports per-seed values, mean and standard deviation, and paired moving-block bootstrap confidence intervals for loss differences. The block duration is 24 hours.

## Literature Results

Reported results from papers with different datasets or protocols are contextual only. Superiority claims must come from reimplemented models on this repository's common benchmark.
