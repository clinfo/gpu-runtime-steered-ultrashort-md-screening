# Method and limitations

This page defines the MD-score calculated by `gpu-shortmd-screening` and the
limits on its scientific and workflow interpretation.

## Prepared-system boundary

Real execution starts from a scientifically prepared GROMACS protein–ligand
system. The package does not generate poses, dock ligands, choose protonation
states, assign charges, infer force fields, parameterize ligands, generate
topologies, select binding sites, or repair molecular structures.

Pose generation and docking are therefore upstream dependencies of this
workflow. Input preparation choices remain part of the scientific provenance
and can affect the resulting trajectories and score.

## Independent-velocity replicas

Each requested replica uses the same prepared pose and resolved protocol with a
different NVT velocity seed. One replica equals one scheduler task. Explicit,
base-derived, or generated seeds are resolved once, stored before execution,
and reused on resume.

These short independent-velocity replicas are not automatically a converged
thermodynamic ensemble.

## PBC reconstruction

The production TPR provides topology and atom masses. The first production
trajectory frame is used as the coordinate reference.

Before RMSD calculation, both the reference and sampled trajectory are
reconstructed as the exact named `Protein_LIG` complex using:

```text
gmx trjconv -pbc cluster
```

The workflow explicitly remaps atom order after extraction. It does not
separately repair broken protein or ligand molecules before `Protein_LIG`
clustering. Prepared inputs must already be molecularly coherent and suitable
for this operation.

## Fitting and ligand RMSD

Each reconstructed frame is fitted to the reference by rotation and
translation of protein `C-alpha` atoms. GROMACS then calculates unweighted RMSD
for ligand atoms that satisfy all of these conditions:

1. membership in the prepared `LIG` index group;
2. the configured `input.ligand_resname`; and
3. TPR mass greater than 2.5 Da.

This is a ligand-heavy-atom measurement after protein fitting. Ligand fitting,
mass-weighted RMSD, a different protein group, missing PBC clustering, or a
different atom selection defines a different metric.

## Units and aggregation

GROMACS XVG values are read explicitly in nanometres and converted to
angstroms. Units are not guessed.

For replica `r`, the replica maximum is:

```text
M_r = max over sampled time (ligand-heavy RMSD in Å)
```

For a completed, unpruned pose with all requested replicas finished:

```text
MD-score = max over all requested replicas (M_r)
```

The calculation is therefore maximum over time and then maximum over replicas.
It is not a mean, median, endpoint, fitted parameter, or population estimate.

The maximum-over-replicas rule is deliberately conservative: the largest
sampled ligand displacement determines the pose score. Increasing trajectory
length, output frequency, or replica count creates more opportunities to
observe a larger excursion.

## Completion and pruning

All requested replicas must complete for an unpruned pose to receive a
completed `md_score_angstrom`. A pruned or otherwise incomplete pose keeps its
observed maxima, status, and trigger information, but receives no completed
MD-score.

Pruning uses the strict condition `observed RMSD > threshold`; equality does
not trigger it. A threshold is target- and protocol-specific and requires
scientific justification. Thresholds used to demonstrate runtime behavior are
not universal screening cutoffs.

### Related runtime-steering work

The design of the pruning and work-stealing mechanisms in this software draws
on the early-termination and parallel-scheduling framework described by Okuno
et al. The paper describes work stealing as a scheduling strategy and identifies
its broader evaluation as future work.

Okuno, S., Kanazawa, N., Koyama, T., Katoh, T., Matsumoto, S., and Okuno, Y.
High-Performance Virtual Screening Based on Parallel Molecular Dynamics
Simulations with Early Termination. In *Proceedings of the 55th International
Conference on Parallel Processing (ICPP '26)*, in press, 2026.
<https://doi.org/10.1145/3832810.3832919>.

## Interpretation

Under one fixed preparation and simulation protocol, a lower MD-score means
less sampled ligand-heavy-atom displacement and therefore greater
short-timescale geometric stability of the starting pose.

MD-score is not:

- binding affinity or a dissociation constant;
- binding free energy;
- residence time or another kinetic constant;
- equilibrium occupancy or proof of convergence;
- a calibrated probability of activity; or
- direct evidence that binding occurs experimentally.

A low value does not establish these claims, and a high value does not by
itself identify why the ligand moved.

## Scientific limitations

- Short trajectories do not establish equilibrium or convergence.
- Preparation, docking, protonation, parameterization, force field, solvent,
  ions, and restraints can change the result.
- Duration, output interval, replica count, and seeds affect the maximum
  statistic and must be reported.
- Tested trajectory lengths, replica counts, GPU models, and pruning thresholds
  are validation examples, not universal scientific choices.
- Scores from different protocols should not be treated as directly
  equivalent without a declared comparison and sensitivity analysis.
- Pruned and completed poses provide different evidence; any downstream policy
  for null scores is a separate analysis choice.

## Workflow limitations

- The validated MD platform is Linux with NVIDIA/CUDA GROMACS 2025.4.
- AMD GPU, multi-node execution, distributed work stealing, and production
  Slurm integration are not validated.
- Functional hardware validation does not establish performance, throughput,
  scaling, or minimum hardware requirements.
- The scheduler validation used logical pose IDs referencing the same prepared
  system; it did not validate scientifically distinct poses in that scenario.
- The bundled historical XVGs and local crosswalk map to the activity-free
  [Zenodo Version 2 reference dataset](https://doi.org/10.5281/zenodo.21835249).
  They validate analysis and units, not identity with newly generated
  trajectories.

For operational details, see [Prepared input](prepared_input.md),
[Runtime control](runtime_control.md), and
[Validated environments](validated_environments.md).
