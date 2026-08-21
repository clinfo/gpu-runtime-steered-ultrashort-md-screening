# Validated environments

The software corresponding to version 0.1.0 was functionally validated on
Linux with CUDA-enabled GROMACS 2025.4 in the environments below.

| Linux environment example | GROMACS | CUDA | Functional scope |
|---|---|---|---|
| 1× NVIDIA RTX A6000 | 2025.4 | 12.2 | One- and five-replica execution, online pruning, and controlled stop/checkpoint resume |
| 3× NVIDIA RTX PRO 6000 Blackwell Max-Q | 2025.4 | 12.8 | Three-GPU scheduling, task stealing, pose-scoped pruning, running-sibling cancellation, and cross-pose redistribution |

These GPU and CUDA combinations are tested examples, not minimum
requirements. The tests cover prepared-system execution, independent-velocity
replica aggregation, `Protein_LIG` reconstruction and ligand-RMSD continuity,
optional pruning, controlled stop and checkpoint resume, round-robin scheduling
with pending-task work stealing, and pose-scoped cancellation while other poses
continue.

This is functional validation, not a performance, throughput, scaling, or
energy-efficiency benchmark, and the short validation trajectories do not
establish convergence for new systems.

A single-replica run completed on a GPU that was also running a resident CUDA
service. This shows that coexistence was possible in the tested setup, but
available memory and interference from other workloads remain system-dependent.

The multi-GPU scheduling tests used two logical pose IDs that referenced the
same prepared system with different seed sets. They tested scheduling, not
scientifically distinct docking poses.

## Not validated

- AMD GPUs or every NVIDIA GPU/CUDA/GROMACS combination;
- multi-node execution or distributed work stealing;
- production Slurm deployment;
- raw receptor/ligand preparation, docking, or topology generation;
- scientific suitability, equilibrium convergence, affinity, or free energy
  for arbitrary prepared systems; or
- prospective experimental outcomes.

Each run records its source revision and environment; validation claims apply
only to the documented software version and scope.

See [Method and limitations](method_and_limitations.md) for scientific scope
and [Running](running.md) for the scheduler model.
