# Prepared input

Real execution starts from a scientifically prepared, runnable GROMACS
protein–ligand system. The package does not generate a pose, repair molecular
structures, choose protonation or force fields, parameterize a ligand, or
create a topology.

## Directory contents

Every docked pose normally has its own self-contained prepared-system
directory. Two manifest entries may share a directory only when they refer to
the identical prepared system; different coordinate poses need different
directories.

| Required artifact | Format | Role |
|---|---|---|
| Starting coordinates | `.gro` | Coordinates supplied to NVT |
| Main topology | `.top` | System topology and include graph |
| Local topology includes | `.itp` | Molecule and restraint definitions |
| Index | `.ndx` | Exact named analysis and coupling groups |
| NVT settings | `.mdp` | Velocity generation and NVT stage |
| NPT settings | `.mdp` | Pressure-equilibration stage |
| Production settings | `.mdp` | Short production trajectory |

For `gpu-shortmd inspect` and a single-pose run, the common configuration
supplies the prepared-system directory and filenames:

```yaml
input:
  prepared_system_dir: prepared/pose-001
  start_structure: start.gro
  topology: topol.top
  index: index.ndx
  ligand_resname: LIG
  fit_group: C-alpha
  ligand_group: LIG
stages:
  nvt: {enabled: true, mdp: mdp/nvt.mdp}
  npt: {enabled: true, mdp: mdp/npt.mdp}
  production: {enabled: true, mdp: mdp/production.mdp}
```

During a screening run, each entry in `poses.yaml` supplies the prepared-system
directory and filenames for that pose:

```yaml
poses:
  - pose_id: TARGET1_CMPD0001_pose01
    prepared_system_dir: prepared/TARGET1_CMPD0001_pose01
    start_structure: start.gro
    topology: topol.top
    index: index.ndx
    ligand_resname: LIG
```

`pose_id` is the project-local identifier used in run outputs.

## Path and include rules

`input.prepared_system_dir` resolves relative to the common configuration file,
and each manifest `prepared_system_dir` resolves relative to the manifest file.
`start_structure`, `topology`, `index`, the stage MDP filenames, and
recursively resolved local topology includes resolve inside that
prepared-system directory and must stay there. Absolute paths are accepted;
`..` traversal or symlinks that let a required input escape the directory are
rejected.

Installed GROMACS force-field includes may resolve through the detected
GROMACS data prefix. Their hashes are recorded, but the installed force field
is not copied into the run bundle.

The software snapshots prepared inputs and records SHA-256 checksums before
execution. Do not edit the source or frozen copies during a run. Resume checks
the stored identities and refuses inputs or dependencies that no longer match.

## Required index groups

The index must contain exactly one non-empty group with each case-sensitive
name:

- `C-alpha`
- `LIG`
- `Protein_LIG`
- `Water_and_ions`

A required name may not occur twice, and a required group may not contain a
duplicate atom index. `C-alpha` and `LIG` must both be subsets of
`Protein_LIG`.

`input.ligand_resname` is an atom-selection input, not descriptive metadata.
It accepts 1–16 ASCII letters, digits, underscores, plus signs, or minus signs.
The generated ligand-heavy group intersects the prepared `LIG` group, the
configured residue name, and atoms whose TPR mass is greater than 2.5 Da. An
empty or inconsistent result fails validation.

## MDP rules

All three stages must be enabled and use the `md` integrator.

- NVT must set `gen_vel = yes`.
- NPT and production must set `gen_vel = no`.
- Production duration divided by `dt` must be an integer number of steps.
- `trajectory.output_interval_ps` must be an integer multiple of `dt` and
  match `nstxout-compressed × dt` in the source production MDP.

For each replica, the software writes normalized MDPs into the run directory,
sets its independent NVT seed, and resolves production `nsteps` and output
frequency. It does not mutate the source MDPs.

## Validate before running

```bash
gpu-shortmd inspect --config run.yaml --json --output inspect_report
gpu-shortmd run --config run.yaml --manifest poses.yaml --dry-run
```

`inspect` checks the common configuration and the pose specified in `input`.
The dry-run checks every pose in the manifest and the full task plan. Neither
command can decide whether preparation, docking, force field, protonation,
solvent, restraints, duration, or sampling choices are scientifically
appropriate. The exact downstream measurement is defined in
[Method and limitations](method_and_limitations.md).
