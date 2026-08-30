# Agent Review Checklist

## Scope
- Did the agent implement only the requested phase?
- Did it add prohibited complexity?

## Data
- Was raw data preserved?
- Were timestamp gaps and exclusions reported?
- Were 1h buckets required to contain 12 bars?

## Leakage
- Are features causal?
- Is split assignment based on target timestamp?
- Are scalers train-only?
- Was the final test untouched?

## Comparison
- Is only one variable changed?
- Are timestamps, horizon, and lookback comparable?
- Are results derived from saved predictions?

## Evidence
- Are commands and exact results shown?
- Are tests relevant and actually run?
- Are manifests and outputs listed?
