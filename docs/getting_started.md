# Getting started

This guide installs the package, then takes a screening work directory from
prepared poses through inspection, a dry-run, and real execution.

## Requirements

Real MD execution requires Linux, Python 3.11 or 3.12, CUDA-enabled GROMACS
2025.4, one or more CUDA-capable NVIDIA GPUs, and one prepared GROMACS system
for every docked pose. Storage depends on pose count, replicas per pose,
trajectory duration, and output interval.

GROMACS, CUDA, and the NVIDIA driver are external dependencies. Check them:

```bash
python3 --version
gmx --version
nvidia-smi -L
```

Check for Python 3.11/3.12, `GROMACS version: 2025.4`, and `GPU support: CUDA`,
and confirm that the GPUs you plan to use are listed. Tested model/CUDA
combinations are examples, not minimum requirements; see
[Validated environments](validated_environments.md).

Analysis-only commands such as `validate-example` and `score` do not launch
GROMACS.

## Install from a source checkout

```bash
git clone https://github.com/clinfo/gpu-runtime-steered-ultrashort-md-screening.git
cd gpu-runtime-steered-ultrashort-md-screening
python3 --version  # Confirm Python 3.11 or 3.12.
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
gpu-shortmd --version
gpu-shortmd --help
```

## Validate the analysis installation

From the repository root:

```bash
gpu-shortmd validate-example \
  examples/egfr_p00533_1xkk_fmm_p2_model4
```

Successful JSON output includes:

```json
{
  "md_score_angstrom": 4.872631,
  "n_replicas_completed": 5,
  "status": "COMPLETED"
}
```

This reads five stored XVG series and checks parsing, unit conversion, and
aggregation. It does not start a trajectory, and a new trajectory is not
expected to reproduce this number. The fixture comes from the
[Zenodo Version 2 reference dataset](https://doi.org/10.5281/zenodo.21835249).

## Create a screening work directory

```bash
cp -R templates/screening my_shortmd_screen
cd my_shortmd_screen
```

The template recommends:

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

The template does not contain empty prepared directories. Create one
self-contained directory per pose and populate it with `start.gro`, `topol.top`
and local includes, `index.ndx`, and the three MDP files under `mdp/`. Review
[Prepared input](prepared_input.md) before using a new system.

## Edit the two YAML files

The screening template uses `run.yaml` for the settings common to every pose
and `poses.yaml` for the pose list.

In `run.yaml`, review at least:

- `run.name` and `run.output_dir`;
- the pose under `input`;
- replica count and the seed policy;
- duration and output interval;
- the three MDP filenames;
- `gromacs.executable`, `ntmpi`, and `ntomp`;
- GPU scheduling; and
- pruning, which is disabled by default.

In `poses.yaml`, write one entry per docked pose. Update each unique `pose_id`,
prepared-system directory, filenames, and ligand residue name. The example
shows multiple poses for one compound, another compound, and another
target/CPI.

## Understand path resolution

- `run.output_dir` and `input.prepared_system_dir` resolve relative to
  `run.yaml`.
- Each manifest `prepared_system_dir` resolves relative to `poses.yaml`.
- `start_structure`, `topology`, `index`, and the stage MDP filenames resolve
  inside the prepared-system directory of each pose.

## Inspect the common configuration

After populating the first prepared-system directory:

```bash
gpu-shortmd inspect --config run.yaml
```

`inspect` checks the common configuration and the pose specified in `input`,
together with the topology/index/MDP contracts, the GROMACS/CUDA/GPU
environment, and the output location and available disk space. It does not
start MD.

## Validate the complete screening plan

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0 \
  --dry-run
```

`--gpu-ids` takes a comma-separated list, so use `0,1,2` to run on three GPUs.

The dry-run checks every pose in the manifest, resolves unique seeds, plans one
task per pose-replica pair, assigns initial GPUs, and writes the resolved
configuration and execution plan without launching MD. Correct every error and
review `execution_plan.json` before continuing.

## Execute

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0
```

Run directories appear below `outputs/` for the template. Do not edit source or
frozen inputs while a run is active. When execution finishes, review:

1. `pose_summary.csv` for pose status and completed MD-score;
2. `replica_summary.csv` for replica status, seeds, GPUs, and maxima; and
3. `run_report.md` for a readable outcome and recorded issues.

See [Outputs](outputs.md) before moving or archiving a run and
[Running](running.md) for task scheduling, one-GPU/three-GPU examples, and the
single-pose smoke-test workflow. For controlled interruption, see
[Runtime control](runtime_control.md).
