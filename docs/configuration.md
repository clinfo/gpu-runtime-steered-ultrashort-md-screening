# Configuration

Configuration files use a strict YAML schema. Unknown keys and invalid values
are rejected. Use
[`templates/screening/run.yaml`](../templates/screening/run.yaml) and
[`templates/screening/poses.yaml`](../templates/screening/poses.yaml) for a
screen, with `configs/config.schema.json` and
`configs/pose-manifest.schema.json` as machine-readable references.

## Common conditions and pose inventory

The screening template uses `run.yaml` for the settings common to every pose
and `poses.yaml` for the pose list.

- Set `input` in `run.yaml` to one pose from the manifest so that
  `gpu-shortmd inspect` can check the shared setup. The template uses the first
  entry.
- Each `poses.yaml` entry gives the prepared-system directory and prepared
  filenames for one docked pose.

Use `--config` and `--manifest` if you choose different filenames. Keeping both
at the top of the screening work directory keeps relative paths reviewable.

## Path resolution

- `run.output_dir` and `input.prepared_system_dir` resolve relative to the
  common config file unless absolute.
- A relative CLI `--output-dir` resolves relative to the common config file.
- Each manifest `prepared_system_dir` resolves relative to the manifest file
  unless absolute.
- `start_structure`, `topology`, `index`, and NVT/NPT/production MDP filenames
  resolve inside the prepared-system directory of the pose.
- During a manifest run, each manifest entry supplies its own prepared-system
  directory and prepared filenames before that pose is inspected.

## 1. Output and run identity

| Key | Accepted value and purpose |
|---|---|
| `schema_version` | Fixed to `1` |
| `run.name` | Optional run and directory label |
| `run.output_dir` | Non-empty output root |
| `run.resume` | Must remain `false`; use `--resume RUN_DIR` |
| `run.overwrite` | Fixed to `false` |

An existing run directory is never overwritten. The screening template writes
new run directories under `outputs/` next to `run.yaml`.

## 2. The pose used by `inspect`

| Key | Accepted value and purpose |
|---|---|
| `input.prepared_system_dir` | Prepared-system directory of one pose |
| `input.start_structure` | Starting `.gro` under that root |
| `input.topology` | Main `.top` under that root |
| `input.index` | Required `.ndx` under that root |
| `input.ligand_resname` | 1–16 ASCII letters, digits, `_`, `+`, or `-` |
| `input.fit_group` | Fixed to `C-alpha` |
| `input.ligand_group` | Fixed to `LIG` |

`inspect` reads the common configuration only, so these paths must already
exist before you inspect. In a manifest run, each pose entry supplies its own
prepared-system directory and these four pose-specific fields.

Prepared-file and index requirements are in
[Prepared input](prepared_input.md).

## 3. Replicas and seeds

| Key | Accepted value and purpose |
|---|---|
| `trajectory.replicas` | Integer of at least 1; default for every pose |
| `trajectory.base_seed` | Optional first positive 32-bit GROMACS seed |
| `trajectory.seeds` | Optional list of one unique seed per replica |

Choose at most one seed mode. A `base_seed` derives consecutive values;
`seeds` supplies every value explicitly. If both are null, seeds are generated
when the run is created and persisted. All resolved seeds must be unique across
every task in the run.

For a many-pose screen, either use the template's generated-seed policy or set
non-overlapping per-pose seeds/base seeds in `poses.yaml`. Reusing one common
non-null base seed for several poses produces duplicates and is rejected.

## 4. Trajectory and stages

| Key | Accepted value and purpose |
|---|---|
| `trajectory.production_time_ns` | Positive production duration in ns |
| `trajectory.output_interval_ps` | Positive compressed-coordinate interval in ps |
| `stages.nvt.enabled` | Must be `true` |
| `stages.nvt.mdp` | NVT MDP filename in each prepared-system directory |
| `stages.npt.enabled` | Must be `true` |
| `stages.npt.mdp` | NPT MDP filename in each prepared-system directory |
| `stages.production.enabled` | Must be `true` |
| `stages.production.mdp` | Production MDP filename in each prepared-system directory |
| `monitoring.poll_interval_seconds` | Positive wall-clock polling interval |

The production duration and output interval must agree exactly with the
production MDP timestep and compressed-output cadence.

## 5. GROMACS resources

| Key | Accepted value and purpose |
|---|---|
| `gromacs.executable` | Non-empty PATH name or executable path |
| `gromacs.ntmpi` | Integer of at least 1 |
| `gromacs.ntomp` | Positive integer or `auto` |
| `gromacs.pin` | Boolean thread-pinning request |
| `gromacs.offload.nonbonded` | `gpu`, `cpu`, or `auto` |
| `gromacs.offload.pme` | `gpu`, `cpu`, or `auto` |
| `gromacs.offload.bonded` | `gpu`, `cpu`, or `auto` |
| `gromacs.offload.update` | `gpu`, `cpu`, or `auto` |
| `gromacs.maxwarn` | Nonnegative integer; default `0` |

With `ntomp: auto`, available CPU affinity/count is divided conservatively
across selected GPU workers and `ntmpi`. An explicit oversubscribing value is
rejected.

## 6. GPU scheduling

| Key | Accepted value and purpose |
|---|---|
| `scheduler.backend` | Fixed to `local` |
| `scheduler.gpu_ids` | `auto` or a non-empty unique list of nonnegative IDs |
| `scheduler.work_stealing` | Boolean pending-task stealing switch |
| `scheduler.tasks_per_gpu` | Fixed to `1` |

CLI `--gpu-ids 0,1,2` overrides only `scheduler.gpu_ids`. Tasks start
round-robin across the selected GPUs. When a GPU becomes idle, it can take a
pending task from another GPU's queue; a task already running stays where it is.

## 7. Pruning

| Key | Accepted value and purpose |
|---|---|
| `pruning.enabled` | Boolean; disabled in the screening template |
| `pruning.threshold_angstrom` | Positive when enabled, otherwise null |
| `pruning.grace_period_seconds` | Nonnegative cooperative-stop grace period |

The trigger is strictly `observed RMSD > threshold`; equality does not trigger.
The threshold is a scientific parameter and requires system-specific
justification. Pruning stops the unfinished replicas of the triggering pose
only; the other poses in the screen continue.

## 8. Restart and logging

| Key | Accepted value and purpose |
|---|---|
| `restart.retry_failed` | Default failed-task retry policy |
| `restart.validate_existing_outputs` | Resume-output validation policy |
| `logging.level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `logging.event_log` | Structured event log switch |
| `logging.console` | Console log switch |

Normal resume does not retry failed tasks. Review
[Runtime control](runtime_control.md) before using `--retry-failed`.

## Fixed analysis definition

These fields identify the calculation and are not alternate modes:

| Key | Required value |
|---|---|
| `rmsd.backend` | `gromacs` |
| `rmsd.input_unit` | `nm` |
| `rmsd.output_unit` | `angstrom` |
| `rmsd.fit_group` | `C-alpha` |
| `rmsd.ligand_group` | `LIG` |
| `rmsd.heavy_atoms_only` | `true` |

See [Method and limitations](method_and_limitations.md).

## Pose manifest and optional overrides

Each manifest entry requires a unique `pose_id`, `prepared_system_dir`,
`start_structure`, `topology`, `index`, and `ligand_resname`:

```yaml
schema_version: 1
poses:
  - pose_id: TARGET1_CMPD0001_pose01
    prepared_system_dir: prepared/TARGET1_CMPD0001_pose01
    start_structure: start.gro
    topology: topol.top
    index: index.ndx
    ligand_resname: LIG
```

Per-pose overrides are optional and limited to trajectory `replicas`,
`base_seed`, `seeds` and pruning `enabled`, `threshold_angstrom`,
`grace_period_seconds`. Keep common conditions in `run.yaml`; use overrides
only when a pose intentionally differs.

## CLI roles

- `inspect --config` checks the common configuration and the pose specified in
  `input`.
- `run --config --manifest --dry-run` checks every pose in the manifest and
  writes the full task plan without MD.
- `--gpu-ids` overrides only the configured GPU list.
- `--output-dir` overrides only the configured output root.
- Omitting `--manifest` runs the pose specified in `input` on its own.

Every real or dry run stores the resolved configuration, manifest, seeds,
input hashes, and execution plan.
