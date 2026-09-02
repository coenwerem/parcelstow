# Candidate Task Authoring Protocol

This protocol defines the evidence required to propose a task for ParcelStow. A candidate task must measure expert and learned-policy success across a speedup-factor grid while preserving its own geometry, observation semantics, phase names, stage outcomes, and terminal failure reasons.

## Mandatory Task Definition

The proposal must specify all of the following before learner evaluation begins.

1. **Robot embodiment and controlled degrees of freedom:** identify the robot, hand, controlled joints, controller rate, command type, joint ordering, limits, and action scaling.
2. **Manipulated object and fixture geometry:** give dimensions, mass and inertial properties, collision geometry, materials, clearances, and fixture poses. State whether the object is free throughout manipulation.
3. **Start and goal distributions:** define every randomized variable, distribution, bound, correlation, and seed procedure for the robot, object, and fixture.
4. **State observation and action:** enumerate observation fields in concatenation order with dimensions, frames, units, noise, clipping, and task-specific slices. Define action semantics without renaming frozen public fields.
5. **Phase set and phase durations:** list the task's own phase names, nominal durations, and transition rules.
6. **Scaled and fixed phases:** mark each phase whose duration is divided by speedup factor `r` and each phase whose duration remains fixed.
7. **Expert construction:** describe the scripted policy, controller, trajectories, grasp selection, feedback, and any model or optimization used at runtime.
8. **Expert-only calibration:** complete calibration before training or evaluating a learner. Record candidate schedules, rejected settings, fixed acceptance criterion, seeds, and results.
9. **Demonstrated speed range:** freeze the minimum and maximum `r` used for demonstration collection before observing learner evaluation results.
10. **Demonstration collection:** state target and accepted episode counts, rate sampling, success filtering, observation/action storage, temporal sampling, seeds, and checkpoint-training inputs.
11. **Physical success predicates:** define success using simulated physical state, contacts, containment, pose, velocity, or settling conditions with units and thresholds.
12. **Stage outcomes:** define task-specific Boolean milestones and when each becomes true. Do not impose ParcelStow stage names on another manipulation sequence.
13. **Terminal failure reasons:** define a mutually interpretable task-specific classification and its precedence when multiple conditions occur.
14. **Matched initial conditions:** use the same per-speed initial-condition draws and seeds for every compared actor, or state and justify a documented exception.
15. **Evaluation grid and episode count:** freeze evaluated speedup factors, episodes per policy-speed pair, evaluation seeds, corruption setting, and jitter before learner results are inspected.
16. **Required record fields:** store task ID, policy, checkpoint, speedup factor, seed, episode, initial and final object pose, task duration, stage outcomes, task success, failure reason and detail, configuration stamp, and task-specific physical diagnostics.
17. **Numerical and physical-integrity tests:** assert record schema, exact released counts, schedule scaling, observation/action dimensions, free-object dynamics, success thresholds, failure labels, matched draws, and malformed-record failure behavior.
18. **Documentation and video evidence:** provide a task specification, installation and evaluation commands, observation/action table, speed range, record map, limitations, and at least one expert and learner episode with accurate captions. Videos do not replace records or tests.
19. **Asset provenance:** identify the source, license, and transformation history for robot, object, fixture, trajectory, and grasp files. Preserve original source names and checksums where licensing permits redistribution.
20. **Admission criteria:** demonstrate a calibrated expert, a precommitted evaluation range, released demonstrations and records, CPU-only numerical reproduction, task-specific simulator tests, a public evaluator route, complete provenance, and documentation that separates observations from interpretations.

## Required Review Package

Submit the task definition, expert calibration record, demonstration summary, policy checkpoints or download manifest entries, episode records, exact-count tests, physical-integrity tests, reproduction command, simulator commands, environment versions, and limitations. The review must be able to trace each number to a source record and each physical predicate to executable code.

## Rejected Task Designs and Procedures

ParcelStow rejects a candidate when any of the following applies:

- the demonstrated range or evaluation grid was chosen after learner results were seen;
- task success depends on an analytical grasp score instead of the simulated physical outcome;
- a welded or fixed manipulated object is presented as grasping;
- ParcelStow stages are renamed and imposed on an unrelated task;
- the expert was not calibrated before learner evaluation;
- the task claim is supported only by videos;
- frozen released files are changed in place;
- learner and expert results use unmatched initial conditions without a predeclared justification;
- observation or action divergence is concealed to claim a common interface.

## Recommended Diagnostics

The following diagnostics improve interpretation but are not mandatory admission requirements unless the task uses them in a claim:

- stage-completion curves over speed;
- contact-set and slip measurements at task-relevant events;
- actuator utilization and target-tracking error;
- object-pose traces and fixture-contact forces;
- expert sensitivity to geometry, friction, mass, and control rate;
- multiple learner seeds and demonstration-count ablations;
- calibrated confidence intervals and matched-pair analyses.

Diagnostics must retain their measurement names and evidence level. A correlation, analytical score, or single video does not establish a physical mechanism.

## Frozen File Policy

After a task enters a versioned release, its geometry, schedules, initial-condition distributions, success predicates, schemas, records, demonstrations, checkpoints, and reported results are immutable. Corrections or protocol changes require new versioned files and an explicit compatibility statement. They must not replace the released files that existing commands and citations resolve.
