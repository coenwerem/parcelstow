# Contributing to ParcelStow

ParcelStow accepts software fixes, documentation corrections, policy integrations, policy results, and candidate benchmark tasks. The immutable `v1.0.0` release and its records define the parcel-insertion study reported in arXiv:2609.01453. Do not modify released task definitions or records in place.

## Contribution Types

- **Bug reports:** include the commit, operating system, Python and Isaac Lab environment, exact command, complete error, and smallest reproduction.
- **Documentation corrections:** identify the code, record, task specification, or external source that supports the correction.
- **New policy results:** submit episode-level records and summaries produced by the public evaluator. State the policy checkpoint, evaluation seed, speed grid, episode count, and demonstrated speed range.
- **New policy integrations:** implement the actor contract in [Policy Interface](docs/POLICY_INTERFACE.md), add a no-Isaac import test, and provide one short evaluation command for each supported task.
- **Candidate benchmark tasks:** follow [Task Authoring](docs/TASK_AUTHORING.md). A task is not part of ParcelStow until its scientific protocol, records, tests, and documentation pass review.
- **Changes to frozen task definitions:** open an issue before writing code. Geometry, phase schedules, initial-condition distributions, observations, actions, success predicates, records, checkpoints, and reported results from a release remain immutable. Corrections require a new versioned artifact; never edit released files in place.

## Local Setup

Install the ParcelStow extension from the repository root using the Isaac Lab Python interpreter:

```bash
python -m pip install -e source/parcelstow
```

Install CPU-only analysis and test dependencies in a separate environment when Isaac Lab is not needed:

```bash
python -m pip install numpy matplotlib pytest ruff torch
```

Download only the demonstrations and ACT checkpoint required for a task:

```bash
python scripts/download_artifacts.py --task parcel
python scripts/download_artifacts.py --task upright
python scripts/download_artifacts.py --task peg
```

## Formatting and Pure Tests

Run formatting checks on every supported Python file changed by the pull request. The repository uses Ruff settings from `pyproject.toml`.

```bash
ruff check <changed-python-files>
python -m pytest tests/ -q
python scripts/reproduce.py all-tasks
```

Pure tests must not import Isaac Lab, start Isaac Sim, require a GPU, or write into released record directories.

## Simulator Tests

Run the three Isaac test groups in separate processes. `--isaac` does not select the upright or peg groups.

```bash
python -m pytest tests/test_parcel_physics.py tests/test_relative_handoff.py --isaac -q
python -m pytest tests/test_upright_physics.py --isaac-upright -q
python -m pytest tests/test_peg_physics.py --isaac-peg -q
```

Policy or task changes must also include short simulator evaluations through `scripts/evaluate.py`. Record the command, environment, duration, pass/fail/skip counts, failure output, and generated files.

## Record Requirements

Evaluation records are append-only JSON Lines. Every episode must identify the task or frozen configuration, policy, speedup factor, seed, logical episode, `initial_condition_id`, initial object pose, task-specific stage outcomes, terminal failure reason, and physical success result. The expert and learner records for a policy-speed comparison must contain the same `initial_condition_id` values and exact initial poses. Preserve task-specific fields: do not rename parcel fields or force unrelated tasks into parcel stage names. Summary rows must report integer successes and episodes for every evaluated policy-speed pair. Custom-policy episode and summary records store the policy object's canonical name in `policy` and its import specification in `actor_spec`.

Do not rewrite, regenerate, truncate, normalize, or replace files under `data/records/` that belong to a release. Add new records under a task- and contribution-specific path. Derived tables and figures belong in an output directory, never beside the frozen source records.

## Pull Request Evidence

A scientific change must include:

1. the exact hypothesis or correction;
2. the code, configuration, and asset provenance;
3. the fixed evaluation grid, episode count, initial-condition law, and seed procedure;
4. expert-only calibration completed before learner evaluation when the task protocol changes;
5. episode-level records, summaries, and reproduction commands;
6. numerical tests for reported counts and physical-integrity tests for success predicates;
7. separate simulator-test results for parcel, upright, and peg when shared code changes;
8. limitations and any divergence from the documented observation or action semantics.

Videos are supporting evidence, not numerical or physical validation. Reviewers must be able to reproduce every reported count from submitted records without Isaac Lab.

## Review and Release Boundaries

Maintainers review software compatibility and scientific validity separately. Acceptance of a pull request does not make a candidate task part of a stable release. A release requires an explicit versioned software and record boundary. Current development on `main` is not a released v2 package.
