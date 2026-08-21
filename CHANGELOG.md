# Changelog

## 0.1.0

- Provide strict inspection and execution for prepared GROMACS
  protein–ligand systems.
- Run independent-velocity NVT, NPT, and production replicas on one or more
  local NVIDIA GPUs.
- Calculate PBC-corrected ligand-heavy-atom RMSD after protein C-alpha fitting
  and aggregate the maximum over time and replicas in angstroms.
- Support YAML multi-pose manifests, one persistent worker per GPU,
  transactional task claims, and pending-task work stealing.
- Provide optional pose-scoped pruning with retained trigger evidence and null
  completed scores for pruned poses.
- Persist seeds, prepared inputs, resolved configuration, environment/source
  provenance, authoritative SQLite state, events, summaries, and checksums.
- Support controlled stop, checkpoint continuation, resume, and explicit retry
  of failed tasks.
- Include an analysis-only historical XVG reference with expected MD-score
  `4.872631 Å`.
- Document functional validation with CUDA GROMACS 2025.4 on one NVIDIA RTX
  A6000 and on three NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, with explicit
  scope qualifications.
- Limit the stable workflow to prepared systems; preparation, docking,
  multi-node scheduling, and production Slurm integration are not included.
