# Running

The main workflow screens many prepared compound–protein interaction poses
from one common configuration and one pose manifest, on a local machine with
one or more GPUs.

## 1. Create the work directory

Copy the screening template and use this layout:

```text
my_shortmd_screen/
├── run.yaml
├── poses.yaml
├── prepared/
│   ├── TARGET1_CMPD0001_pose01/
│   ├── TARGET1_CMPD0001_pose02/
│   ├── TARGET1_CMPD0002_pose01/
│   └── TARGET2_CMPD0042_pose01/
└── outputs/
```

```bash
cp -R templates/screening my_shortmd_screen
cd my_shortmd_screen
```

The template contains the YAML files, not the prepared directories. Create every
listed directory and populate it with its own runnable GROMACS system.

## 2. Treat one manifest entry as one docked pose

Each `poses.yaml` entry is one geometrical docked pose. Several poses of one
compound are separate entries and normally use separate prepared-system
directories. Two entries may share a directory only when they deliberately refer
to the identical prepared system.

The template inventory represents:

- two poses for `TARGET1_CMPD0001`;
- one pose for `TARGET1_CMPD0002`; and
- one pose for the second target/CPI, `TARGET2_CMPD0042`.

These are generic project-local identifiers, not external database IDs or
activity annotations.

## 3. Calculate the task count

The scheduler creates one task for each pose-replica pair:

```text
total tasks = number of manifest poses × replicas per pose
```

Four poses with five replicas each create 20 tasks. A per-pose replica override
changes the count for that pose only.

## 4. Edit common conditions in `run.yaml`

`run.yaml` defines the run name, output location, replicas and seeds,
trajectory settings, GROMACS resources, GPU scheduling, pruning, and
restart/logging behavior. Set `input` to one pose from the manifest; the
template uses the first entry.

`run.output_dir` and `input.prepared_system_dir` resolve from the config file,
and stage MDP filenames resolve inside the prepared-system directory. Review the
complete field order in [Configuration](configuration.md).

## 5. Edit the manifest in `poses.yaml`

```yaml
schema_version: 1
poses:
  - pose_id: TARGET1_CMPD0001_pose01
    prepared_system_dir: prepared/TARGET1_CMPD0001_pose01
    start_structure: start.gro
    topology: topol.top
    index: index.ndx
    ligand_resname: LIG

  - pose_id: TARGET1_CMPD0001_pose02
    prepared_system_dir: prepared/TARGET1_CMPD0001_pose02
    start_structure: start.gro
    topology: topol.top
    index: index.ndx
    ligand_resname: LIG
```

Every `pose_id` must be unique. Each relative `prepared_system_dir` resolves
from the manifest directory, and prepared filenames and common MDP filenames
resolve inside that prepared-system directory. Optional overrides are limited to
replica/seed and pruning settings.

All resolved velocity seeds must be unique across the entire screen. The
template leaves both common seed fields null, so seeds are generated when the
run is created and then persisted. If you need deterministic seeds, give the
poses non-overlapping per-pose seed lists or base seeds.

## 6. Inspect, dry-run, then execute

Check the common configuration and the pose specified in `input`:

```bash
gpu-shortmd inspect --config run.yaml
```

`inspect` checks the configuration, that pose's files, the topology/index/MDP
contracts, external GROMACS/CUDA/GPU availability, disk space, and output
filesystem behavior. It does not launch MD.

Next check every pose in the manifest and write the full task plan:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0,1,2 \
  --dry-run
```

The dry-run snapshots inputs, resolves seeds and CPU threads, plans every task,
and writes `execution_plan.json` without launching GROMACS. Review the plan,
then execute:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0,1,2
```

## 7. Understand initial GPU assignment

Each selected GPU runs one task at a time, because `tasks_per_gpu` is fixed to
`1`. Tasks start round-robin. With GPUs `0,1,2`:

```text
task 1 → GPU 0
task 2 → GPU 1
task 3 → GPU 2
task 4 → GPU 0
```

A task stays on the GPU that started it.

## 8. Five replicas on one or three GPUs

To run all five replicas of each pose on one GPU:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0
```

That GPU runs the tasks one after another.

To distribute the same pose × five-replica plan across three GPUs:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0,1,2
```

The same set of replica tasks is run; only the GPU assignments change.
`gromacs.ntomp: auto` divides the available CPU allocation conservatively across
the selected GPUs and `ntmpi`.

## 9. Pending-task work stealing

With `scheduler.work_stealing: true`, a GPU that becomes idle can take a pending
task from another GPU's queue. A task already running never moves.

This scheduler runs on one machine. It does not distribute work across nodes or
submit to Slurm.

## 10. Pruning and results

When one replica exceeds the threshold, the software prunes that pose, stops its
running siblings, and skips its pending siblings. Other poses continue.

Use `pose_summary.csv` for pose outcomes and `replica_summary.csv` for the seed,
GPU, status, and maximum of every replica. See [Outputs](outputs.md),
[Runtime control](runtime_control.md), and
[Method and limitations](method_and_limitations.md).

## 11. Single-pose run

For a smoke test or a focused follow-up on a single pose, omit `--manifest`:

```bash
gpu-shortmd run --config run.yaml --gpu-ids 0 --dry-run
gpu-shortmd run --config run.yaml --gpu-ids 0
```

The run then uses the pose specified in `input`. This is supported, but it is
not the main screening workflow.
