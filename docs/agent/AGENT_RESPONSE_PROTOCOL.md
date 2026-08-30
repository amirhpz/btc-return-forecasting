# Agent Response Protocol

Every implementation response must use this structure.

```markdown
## Task Summary
- Phase:
- Experiment:
- Objective:
- Files inspected:
- Files modified:
- Files created:

## Assumptions

## Implementation Details

## Commands Run
| Command | Result | Evidence |
|---|---|---|

## Tests
| Test | Result | Notes |
|---|---|---|

## Outputs Produced

## Data Exclusions or Repairs

## Risks / Open Issues
| Issue | Severity | Impact | Required action |
|---|---:|---|---|

## Scope Compliance
- Future-phase work added: YES/NO
- Test set accessed: YES/NO
- Raw data modified: YES/NO
- Scaling fit outside train: YES/NO

## Phase Decision
COMPLETE / COMPLETE WITH CONDITIONS / NOT COMPLETE

## Next Permitted Step
```

Statements such as `done`, `should work`, `probably`, or `reproduced the paper` are not acceptable without evidence.
