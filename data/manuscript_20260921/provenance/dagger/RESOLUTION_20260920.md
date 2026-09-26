# Resolution of the monitor-predicate validation stop

The user accepted the validation stop and identified its cause on 2026-09-20:
the validator assumed an ordered stage chain that the peg monitor does not
define. This report supersedes the stage-validity conclusion in STOP_REPORT.md.

The peg monitor independently latches physical predicates. Alignment requires
position and orientation tolerances while held; insertion requires containment
depth during insertion and does not require earlier alignment or acquisition.
The validator now checks only these implemented implications:

- `lifted_clear -> acquired`
- `reoriented_upright -> acquired`
- `aligned -> acquired`
- `released -> inserted`
- `settled -> released`
- `task_success -> inserted and released and settled`

The upright checks likewise follow its code: lifted, reoriented, and placed
imply acquired; released implies placed; settled implies released; success
implies placed, released, and settled. Neither task uses a universal adjacent
stage chain, at the episode or aggregate-count level.

The validation category is `monitor_predicate_implication`. The four formerly
flagged peg records remain unchanged and are preserved as valid regression
cases in monitor_predicate_regression_cases.json, with source paths, line
numbers, raw-line hashes, and all predicate values. The old validation result
is retained as validation_adjacent_stage_assumption.json.

Complete validation from the 40 frozen inputs passes: 12,800 episode records,
64 task/actor/rate conditions, 102,400 direct predicate implication checks,
all pairing/metadata/count/checkpoint checks, and all four regression cases.
No frozen records, summaries, checkpoints, monitor code, or evaluation protocol
were modified. This resolves a validator-specification mismatch, not record
corruption, and does not change any recorded task outcome.

The DAgger implementation audit follows this passing validation. Scientific
comparisons remain gated on the implementation audit as well.
