# Reproducibility Checklist

Before accepting a run:

- [ ] Raw file checksum recorded.
- [ ] Raw data unchanged.
- [ ] UTC timestamp policy verified.
- [ ] Missing and duplicate timestamps reported.
- [ ] Resampling completeness reported.
- [ ] Target formula and horizon recorded.
- [ ] Lookback duration and bar count recorded.
- [ ] Exact feature names exported.
- [ ] Split version and boundaries recorded.
- [ ] Target does not cross split boundary.
- [ ] Feature scaler fit on train only.
- [ ] Target scaler fit on train only.
- [ ] Seed recorded.
- [ ] Git commit and dirty status recorded.
- [ ] Device and package versions recorded.
- [ ] Predictions include anchor and target timestamps.
- [ ] Metrics can be recomputed from saved predictions.
- [ ] Test access policy respected.
- [ ] Excluded rows and reasons recorded.
