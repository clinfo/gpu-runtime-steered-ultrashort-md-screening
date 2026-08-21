# Troubleshooting

## Start with inspection

Run inspection before attempting MD:

```bash
gpu-shortmd inspect --config run.yaml --json --output inspect_report
```

Read the saved report rather than only the last console line. Inspection checks
the configuration, prepared inputs, and execution environment before MD starts.
If the run fails after launch, use `run_report.md` and the stage logs to
diagnose it.

## `gpu-shortmd` is not found

Activate the environment where the package was installed and confirm:

```bash
python -m pip show gpu-shortmd-screening
python -m gpu_shortmd --help
```

If `python -m gpu_shortmd` works but `gpu-shortmd` does not, check that the
environment's `bin` directory is on `PATH`. Avoid installing into one Python
environment and running from another.

## GROMACS executable is missing or incompatible

Check the exact executable configured in `gromacs.executable`:

```bash
gmx --version
```

MD execution requires a CUDA-enabled GROMACS build; the tested version is
2025.4. The package does not install, rebuild, or switch GROMACS versions. A
CPU build can help with separate GROMACS diagnostics, but it is not the tested
execution environment.

## GPU visibility or offload preflight fails

Confirm that the GPUs you selected are visible to the current shell and that
the GROMACS build supports the requested offload modes. `audit/preflight.json`
and the inspect report show the detected executable, version, CUDA status,
visible GPUs, and the failure detail.

A tested GPU model does not define a memory requirement for your system. Other
workloads on the same GPU can consume memory and disturb the run.

## Configuration fails before inspection

The schema is strict. Common causes include:

- an unknown key or YAML scalar with the wrong type;
- both `base_seed` and `seeds` being set;
- a seed list length different from `replicas`;
- duplicate seeds across poses;
- pruning enabled without a positive threshold;
- pruning disabled with a non-null threshold;
- duplicate or negative GPU IDs; or
- `tasks_per_gpu` other than `1`.

Compare with `configs/default.yaml` and the JSON schemas rather than adding an
unrecognized field.

## Prepared-input validation fails

Check that all configured files and local topology includes remain under the
pose's `prepared_system_dir`. The index must contain exactly one non-empty
`C-alpha`, `LIG`, `Protein_LIG`, and `Water_and_ions` group. `C-alpha` and
`LIG` must be subsets of `Protein_LIG`, without duplicate atom indices.

Also confirm that `ligand_resname` matches the intended atoms and leaves a
non-empty ligand-heavy group after the TPR `mass > 2.5 Da` selection.

## MDP validation fails

All stages must be enabled and use the `md` integrator. NVT must generate
velocities; NPT and production must not. Production duration/output interval
must resolve to integer steps, and `nstxout-compressed × dt` in the source MDP
must equal `trajectory.output_interval_ps`.

Correct the source MDP yourself, then run inspection again. The software does
not repair an inconsistent scientific input.

## Resume refuses to start

Resume refuses to start when it finds a process that may still be running, so
the same work is not launched twice. Stop the run:

```bash
gpu-shortmd stop RUN_DIR
```

Wait until the process has stopped, then run `gpu-shortmd run --resume RUN_DIR`
again. If the refusal repeats, read the CLI message and the run logs. Do not
edit `state.sqlite3` by hand.

## A task is `FAILED` after resume

Inspect the task logs and reported issue, correct the external problem, then
explicitly opt in:

```bash
gpu-shortmd run --resume RUN_DIR --retry-failed
```

Resume does not retry failed tasks on its own, and tasks under a pruned pose
are never requeued.

## The pose MD-score is null

A null score is expected when the pose was pruned or not all of its requested
replicas completed. Check `pose_summary.csv`, `replica_summary.csv`, and the
run logs for the reason. `observed_max_rmsd_angstrom` may still be present, but
it is not the pose MD-score.

## The historical reference passes but a new trajectory differs

`validate-example` checks the analysis definition against stored XVG values, so
it says nothing about the number a new trajectory will produce. Initial
velocities, preparation, the GROMACS build, the hardware, and stochastic
evolution all change the result.

## Asking for help

Keep the complete run directory and share `run_report.md` together with the
relevant log files. The resolved configuration, the preflight report, and the
status summaries help as well. Remove private paths and infrastructure details
before sharing anything.
