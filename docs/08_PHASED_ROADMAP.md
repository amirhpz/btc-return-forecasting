# Phased Roadmap

## Phase 00: Foundation Scaffold

Deliverables: repository structure, uv environment, configs, contracts, smoke tests.

Gate: `uv run pytest` and `uv run btc-forecast validate-configs` pass.

## Phase 01: E00 Data Sanity and Freeze

Implement loading, validation, 1h resampling, manifests, valid anchors, and frozen split boundaries.

Gate: no unresolved critical data issue; boundary and target alignment tests pass.

## Phase 02: E01-E02 Naive and Linear Baselines

Implement F0, naive suite, Ridge, train-only scaling, and prediction export.

Gate: metrics reproduce from saved artifacts; zero-return baseline is correct.

## Phase 03: E03-E05 Deep Baselines

Implement LSTM, CNN, and CNN-LSTM with shape tests and tiny-batch overfit test.

Gate: training is deterministic enough for debugging; no leakage; outputs complete.

## Phase 04: E06-E07 Feature-Set Comparison

Compare F0, F1, and only then the frozen F2 catalog. Change no other variable.

Gate: select a candidate using validation data only and record the decision.

## Phase 05: E08-E09 Confirmation

Run three seeds on the frozen 1h candidate, then transfer the same feature concept and model to 5m validation. Direct comparison uses the hourly grid.

Gate: code and configs are frozen for final testing.

## Phase 06: E10 Final Test

Open the final test once. Run the predeclared 1h and 5m experiments, export predictions, aggregate seeds, and bootstrap intervals.

Gate: no post-test tuning. Any later change creates a new study version with a new untouched test period.

## Later Work, Not Part of v0.1

Trading evaluation, classification, attention, multi-asset learning, XAI, and evolutionary optimization require a separate approved scope.
