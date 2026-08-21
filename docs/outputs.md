# Outputs

Every dry-run or execution creates a structured run directory. Which files
appear depends on how far the run progressed. Do not commit raw run outputs to
the source repository.

## What to inspect first

1. `pose_summary.csv`: pose status, replica counts, MD-score, and the pruning
   trigger fields.
2. `replica_summary.csv`: seed, assigned GPU, resolved CPU threads, status,
   progress, and maximum RMSD for every replica.
3. `run_report.md`: a readable summary of the run status and pose-level results.

A completed, unpruned pose has a numeric `md_score_angstrom`: the maximum of
its replica maxima. A pruned or incomplete pose has `md_score_angstrom: null`.
`observed_max_rmsd_angstrom` is the largest value seen in the trajectories that
did run, so read it as an observation, not as a pose MD-score.

## Typical output tree

```text
RUN_DIR/
├── state.sqlite3
├── resolved_config.yaml
├── resolved_pose_manifest.yaml
├── input_manifest.json
├── environment.json
├── execution_plan.json
├── pose_summary.csv
├── replica_summary.csv
├── artifact_manifest.csv
├── checksums.sha256
├── run_report.md
├── inputs/
├── audit/
│   └── preflight.json
├── logs/
└── poses/
    └── POSE_DIRECTORY/
        └── replica_XX/
            ├── resolved_mdp/
            ├── nvt/
            ├── npt/
            ├── production/
            ├── rmsd_groups.ndx
            ├── rmsd_time_series_nm.xvg
            └── rmsd_time_series_angstrom.csv
```

`inputs/` holds the frozen prepared-system inputs. Each replica directory holds
its resolved MDPs, the GROMACS stage files, and the ligand-RMSD series. For the
exact inventory a run generated, read `artifact_manifest.csv`.

## Internal state

`state.sqlite3` is the internal workflow state. Do not edit it. Use the CSV,
JSON, Markdown, and log files to inspect a run, and the CLI to stop and resume
it.

## Configuration and provenance

- `resolved_config.yaml` freezes the effective common configuration.
- `resolved_pose_manifest.yaml` freezes normalized pose definitions.
- `input_manifest.json` records snapshotted inputs and checksums.
- `environment.json` records runtime, GPU/GROMACS, scheduler, and detected
  source provenance.
- `execution_plan.json` records tasks, seeds, initial GPUs, stages, threads,
  and hashes.
- `audit/preflight.json` preserves the preflight result.

The validation scope of the recorded environment is summarized in
[Validated environments](validated_environments.md).

## Pose and replica status

Pose summaries include requested and completed replica counts, status,
`observed_max_rmsd_angstrom`, `md_score_angstrom`, and the pruning trigger
fields. Replica summaries record each replica's status, velocity seed, GPU,
resolved `ntomp`, stage reached, completed trajectory time, maximum RMSD,
whether it triggered pruning, exit code, and start/finish times. Error details
are recorded in the event and log files.

Read replica status together with pose status. Under a pruned pose, for
example, the skipped and interrupted siblings explain why that pose has no
MD-score; they are not completed replicas.

## Integrity and archiving

`artifact_manifest.csv` lists the artifacts and `checksums.sha256` records
their hashes. Resume verifies existing outputs before reusing them.

Before copying a run directory, make sure no run process is still writing to
it. Use a controlled stop if the run is still active. Keep the whole directory
if you may resume the run or ask for help: the configuration, seeds, input
hashes, environment, logs, checkpoints, and RMSD series are all needed to
interpret it.

The score definition and its scientific limits are in
[Method and limitations](method_and_limitations.md).
