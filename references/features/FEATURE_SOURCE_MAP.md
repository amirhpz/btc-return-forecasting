# Feature Source Map

Focused audit of the benchmark feature configs against the imported local feature
sources and documentation. No implementation equivalence is assumed from a similar
name.

## F0 Readiness

F0 configures 10 features. Exact matching source definitions: 0. Exact matching
documentation records: 0. Unresolved: 10 (2 ambiguous, 8 missing).

| Feature | Exact formula or source implementation | Required raw columns | Parameters / periods | Minimum lookback / warm-up | Causality | Explicit missing / warm-up behavior | Source / documentation | Code and documentation agree | Status |
|---|---|---|---|---|---|---|---|---|---|
| `log_return_1h` | Not defined | Not defined | `1h` appears only in the name | Not defined | Unverifiable; timestamp alignment is not defined | Not defined | No exact source or document | N/A | MISSING |
| `log_return_3h` | Not defined | Not defined | `3h` appears only in the name | Not defined | Unverifiable; timestamp alignment is not defined | Not defined | No exact source or document | N/A | MISSING |
| `log_return_6h` | Not defined | Not defined | `6h` appears only in the name | Not defined | Unverifiable; timestamp alignment is not defined | Not defined | No exact source or document | N/A | MISSING |
| `candle_log_return` | Not defined | Not defined | Not defined | Not defined | Unverifiable; current-versus-completed-bar alignment is not defined | Not defined | No exact source or document | N/A | MISSING |
| `high_low_range_pct` | Not defined | Not defined | Not defined | Not defined | Unverifiable; denominator and row alignment are not defined | Not defined | No exact source or document | N/A | MISSING |
| `close_position_in_range` | No exact definition. `close_location_value` is related but has a different name, signed output, flat-candle rule, and bar-open `t-1` alignment. | Not defined for F0 | Not defined | Not defined | Unverifiable for F0 | Not defined for F0 | `source/close_location_value.py`; `docs/close_location_value.yaml` | Related source/docs agree internally; they do not define F0 | AMBIGUOUS |
| `rolling_volatility_6h` | Not defined. `realized_vol_20` is a non-matching 20-bar population-standard-deviation feature. | Not defined | `6h` appears only in the name; estimator and `ddof` are undefined | Not defined | Unverifiable; window endpoint is not defined | Not defined | No exact source or document | N/A | MISSING |
| `rolling_volatility_24h` | Not defined. `realized_vol_20` is a non-matching 20-bar population-standard-deviation feature. | Not defined | `24h` appears only in the name; estimator and `ddof` are undefined | Not defined | Unverifiable; window endpoint is not defined | Not defined | No exact source or document | N/A | MISSING |
| `log_volume_change_1h` | Not defined | Not defined | `1h` appears only in the name | Not defined | Unverifiable; timestamp alignment and zero-volume handling are not defined | Not defined | No exact source or document | N/A | MISSING |
| `relative_volume_24h` | No exact definition. Imported sources contain several non-equivalent volume ratios (`vol_over_ema20`, `vol_over_median20`, `vol_ratio_20_100`) with different periods and denominators. | Not defined for F0 | `24h` appears only in the name; baseline statistic is undefined | Not defined | Unverifiable for F0 | Not defined for F0 | Related files under `source/` and `docs/`; no exact source or document | Related pairs agree internally; none defines F0 | AMBIGUOUS |

The only benchmark-wide constraints are that feature windows represent real-time
durations, values must be causal, and unexplained warm-up rows must not be silently
dropped. Those constraints do not supply the missing formulas or row alignment.

## F1/F2 Coverage

### F1

- Configured features: 23 total (F0's 10 plus 13 additions)
- Exact matching imported source definitions: 0
- Exact matching imported documentation records: 0
- Ambiguous or missing: 23
- Unresolved names: `log_return_1h`, `log_return_3h`, `log_return_6h`, `candle_log_return`, `high_low_range_pct`, `close_position_in_range`, `rolling_volatility_6h`, `rolling_volatility_24h`, `log_volume_change_1h`, `relative_volume_24h`, `log_return_12h`, `log_return_24h`, `ema_distance_6h`, `ema_distance_24h`, `atr_pct_14h`, `rsi_14h`, `bollinger_width_20h`, `bollinger_position_20h`, `macd_histogram_norm_12h_26h_9h`, `upper_wick_ratio`, `lower_wick_ratio`, `body_to_range_ratio`, `volume_zscore_24h`

### F2

- Configured benchmark catalog entries: 0; expected count: 52
- Imported catalog source definitions: 52
- Imported per-feature documentation records: 52
- Ambiguous or missing from the benchmark catalog: 52 (all imported entries remain unpopulated in `configs/features/full_52_catalog.csv`)
- Unresolved names: `absret_ema_ratio_20_100`, `amihud_illiquidity_20`, `atr_pct_14`, `atr_ratio_14_63`, `band_bb_percB_20_2`, `bb_excursion_20_2`, `bb_width_rel_20`, `bearish_engulf_score`, `body_signed_to_tr`, `body_to_tr`, `breakdown_strength_20`, `breakout_strength_20`, `bullish_engulf_score`, `cand_up_down_vol_ratio_20`, `channel_pos_20`, `close_location_value`, `dmi_balance_14`, `dollar_vol_rel_20`, `downside_semivol_20`, `drawdown_from_peak_60`, `efficiency_ratio_20`, `ema_gap_atr_20`, `ema_slope_atr_20_5`, `inside_bar_compression`, `log_range_over_vol_100`, `lower_wick_to_tr`, `macd_hist_atr`, `mom_stoch_rsi_14_14_3`, `mom_tl_break_bull_30`, `open_gap_atr_14`, `outside_bar_expansion`, `parkinson_vol_20`, `range_compression_20_100`, `realized_vol_20`, `ret_autocorr_1_30`, `ret_vol_corr_30`, `return_skew_30`, `roc_10`, `rsi_centered_14`, `rsi_div_persistence`, `rsi_hidden_div_flag`, `sign_flip_rate_20`, `tr_to_atr_14`, `up_close_ratio_5`, `upper_wick_to_tr`, `upside_semivol_20`, `vol_of_vol_ratio_20`, `vol_over_ema20`, `vol_over_median20`, `vol_ratio_20_100`, `vol_regime_pct_120`, `volume_percentile_60`

The imported F2 catalog identifies itself as `binance_ohlcv_feature_set` version
`2.0.0`, status `validated`, and records hashes for 52 Python/config pairs. The
benchmark catalog is still header-only, so the imported inventory is not yet a
frozen benchmark definition.

## Provenance / License

- The imported files refer to modules under `features.*` and configs under
  `feature_configs/`, indicating they came from a differently structured project.
- Comments reference MetaTrader/MT5 compatibility and, for ATR, MetaQuotes behavior
  and an MQL5 `CopyBuffer` alignment. This is behavioral provenance, not author or
  project attribution.
- No author attribution, original project name, copyright notice, or license notice
  is present in the audited material.
- No file explicitly states that its code was copied from a third party. Standard
  indicator names and MetaTrader compatibility comments are not treated as evidence
  of copied code.

## E02 Gate

**BLOCKED.** F0 is not fully reproducible, and E02 cannot proceed without inventing
feature formulas, timing alignment, denominator/estimator choices, and warm-up
behavior. No F0 name is inherently future-looking, but causality cannot be verified
until its exact row alignment is frozen. The imported material is sufficient to
audit the 52 imported F2 definitions themselves; it is not sufficient to establish
exact F0/F1 equivalence or to activate F2 in the benchmark catalog.
