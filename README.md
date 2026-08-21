# GPU Runtime-Steered Ultrashort MD Screening

Run independent short-MD replicas for many prepared protein–ligand poses on one
or more NVIDIA GPUs. You set the run conditions once and list one entry per
docked pose in a manifest.

The Python distribution is named `gpu-shortmd-screening`, and its command is
`gpu-shortmd`.

[ChemRxiv preprint](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006352/v1)
· [Getting started](docs/getting_started.md) · [Method and limitations](docs/method_and_limitations.md)

Reference dataset: <https://doi.org/10.5281/zenodo.21835249>

## Requirements

| Requirement | Supported or required value |
|---|---|
| OS | Linux |
| Python | 3.11 or 3.12 |
| MD engine | CUDA-enabled GROMACS 2025.4 (tested release) |
| GPU | One or more CUDA-capable NVIDIA GPUs |
| Input | One prepared GROMACS system per docked pose |
| Storage | Depends on pose count, replicas, duration, and output interval |

GROMACS, CUDA, and the NVIDIA driver are external dependencies. Check them
before configuring a screen:

```bash
python3 --version
gmx --version
nvidia-smi -L
```

Confirm Python 3.11/3.12, `GROMACS version: 2025.4`, `GPU support: CUDA`, and
that the selected NVIDIA GPUs are visible. Validated GPU examples are listed
separately in [Validated environments](docs/validated_environments.md).

## Install

```bash
git clone https://github.com/clinfo/gpu-runtime-steered-ultrashort-md-screening.git
cd gpu-runtime-steered-ultrashort-md-screening
python3 --version  # Confirm Python 3.11 or 3.12.
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
gpu-shortmd --version
```

Check the installation with the bundled reference example:

```bash
gpu-shortmd validate-example \
  examples/egfr_p00533_1xkk_fmm_p2_model4
```

Successful output contains `"md_score_angstrom": 4.872631`. The check reads
stored XVG files and does not launch GROMACS.

## Quick start: screen multiple poses

Copy the complete screening template to a working directory:

```bash
cp -R templates/screening my_shortmd_screen
cd my_shortmd_screen
```

Populate each directory under `prepared/` with one self-contained GROMACS
system; the template does not supply them. The layout and required files are in
the [screening template guide](templates/screening/README.md).

The screening template uses `run.yaml` for the settings common to every pose
and `poses.yaml` for the pose list.

Edit trajectory, replicas, resources, output, and pruning policy in `run.yaml`.
Then list one entry per docked pose in `poses.yaml`, each with the relative path
to its prepared-system directory.

Once the prepared systems are in place, use `inspect` to check the common
configuration and the pose specified in `input`:

```bash
gpu-shortmd inspect --config run.yaml
```

Then use a dry-run to check every pose in the manifest and the planned tasks
without starting MD. Use `0` for one GPU, or replace it with a comma-separated
list such as `0,1,2` to use multiple GPUs:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0 \
  --dry-run
```

Review the generated inputs, seeds, task count, GPU assignment, and pruning
state. Then execute the same plan without `--dry-run`:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0
```

Relative paths in `poses.yaml` resolve from the manifest directory, and
`run.output_dir` resolves from the configuration directory, so the template
writes run directories below `my_shortmd_screen/outputs/`.
[Configuration](docs/configuration.md) gives the full path rules.

Start with `pose_summary.csv`, `replica_summary.csv`, and `run_report.md` in
the new run directory. See [Running](docs/running.md) for scheduling details.

## Prepared systems

Each prepared-system directory must already be runnable by GROMACS and
contain:

| Required input | Typical path | Purpose |
|---|---|---|
| Starting coordinates | `start.gro` | Coordinates for NVT |
| Topology and local includes | `topol.top`, `*.itp` | Prepared system definition |
| Named index groups | `index.ndx` | Fitting, ligand, complex, solvent groups |
| Stage settings | `mdp/nvt.mdp`, `mdp/npt.mdp`, `mdp/production.mdp` | NVT, NPT, production |

The index must contain exactly one non-empty `C-alpha`, `LIG`, `Protein_LIG`,
and `Water_and_ions` group. Preparation, parameterization, repair, and docking
happen outside this package. See [Prepared input](docs/prepared_input.md).

## Tasks and GPUs

The scheduler creates one task for each pose-replica pair. Four poses with five
replicas each create 20 tasks. Tasks are initially assigned to the selected
GPUs in round-robin order. When a GPU becomes idle, it can take a pending task
from another GPU's queue; a task already running stays where it is.

Pruning is disabled in the template. When it is enabled and one replica exceeds
the threshold, the software prunes that pose and the other poses continue. The
threshold requires scientific justification; see
[Runtime control](docs/runtime_control.md).

## Single pose: smoke test or focused follow-up

Omit `--manifest` to run the pose specified in `input` on its own:

```bash
gpu-shortmd run --config run.yaml --gpu-ids 0 --dry-run
gpu-shortmd run --config run.yaml --gpu-ids 0
```

Use this for a smoke test or a focused check of one pose, not for a screen.

## Stop and resume

```bash
gpu-shortmd stop RUN_DIR
gpu-shortmd run --resume RUN_DIR
```

Use `--retry-failed` only after fixing the external cause of the failure. See
[Runtime control](docs/runtime_control.md).

## Scientific scope

Lower MD-score means greater short-timescale pose stability under this
protocol. It is not affinity, free energy, or direct-binding proof. Protocol
settings and pruning thresholds depend on the system and the purpose.

See [Outputs](docs/outputs.md) and
[Method and limitations](docs/method_and_limitations.md) before interpreting
or comparing runs.

## Citation and license

For the method and software, cite Kanazawa et al., *Runtime-steered ultrashort
molecular dynamics enables million-pose protein–ligand screening*, ChemRxiv
(2026), <https://doi.org/10.26434/chemrxiv.15006352/v1>. Machine-readable
software/method metadata are in [CITATION.cff](CITATION.cff).

When using the released reference data, cite: Kanazawa, N., Asano, J., & Matsumoto, S. (2026). Six-target docking poses and MD-scores for runtime-steered ultrashort molecular dynamics (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21835249

Code is MIT licensed under [LICENSE](LICENSE): Copyright (c) 2026 Kyoto
University. Project-generated example/reference data are CC BY 4.0 under the
[data-license notice](docs/legal/data_license.md); underlying PDB data remain
under their source terms. [Contributing](.github/CONTRIBUTING.md)
· [Security](.github/SECURITY.md) · [Code of conduct](.github/CODE_OF_CONDUCT.md)
· [Third-party notices](docs/legal/third_party_notices.md)

## Documentation

- [Getting started](docs/getting_started.md)
- [Prepared input](docs/prepared_input.md)
- [Configuration](docs/configuration.md)
- [Screening and execution](docs/running.md)
- [Pruning, stop, and resume](docs/runtime_control.md)
- [Outputs](docs/outputs.md)
- [Method and limitations](docs/method_and_limitations.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Validated environments](docs/validated_environments.md)
