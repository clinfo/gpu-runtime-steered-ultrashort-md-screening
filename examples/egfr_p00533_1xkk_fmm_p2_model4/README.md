# Bundled computational reference example

This directory holds a reference fixture for pose `P00533_EGFR|p2|model4`,
which is public pose index `00753` in the
[Zenodo Version 2 reference dataset](https://doi.org/10.5281/zenodo.21835249).

## Check the analysis path

From the repository root:

```bash
gpu-shortmd validate-example \
  examples/egfr_p00533_1xkk_fmm_p2_model4
```

The JSON output includes:

```json
{
  "md_score_angstrom": 4.872631,
  "n_replicas_completed": 5,
  "status": "COMPLETED"
}
```

The five RMSD XVG files in `reference/` are in nm and hold the
ligand-heavy-atom RMSD after least-squares fitting to protein Cα atoms. Their
replica maxima are:

```text
replica 01: 4.872631 Å
replica 02: 2.629156 Å
replica 03: 2.638972 Å
replica 04: 2.343307 Å
replica 05: 2.827962 Å
```

The pose-level maximum is therefore `4.872631 Å`.

## Score XVGs directly

Repeat `--xvg` for every complete replica. For example:

```bash
gpu-shortmd score \
  --xvg examples/egfr_p00533_1xkk_fmm_p2_model4/reference/p2_model4_rmsd_replica_01_nm.xvg \
  --xvg examples/egfr_p00533_1xkk_fmm_p2_model4/reference/p2_model4_rmsd_replica_02_nm.xvg \
  --xvg examples/egfr_p00533_1xkk_fmm_p2_model4/reference/p2_model4_rmsd_replica_03_nm.xvg \
  --xvg examples/egfr_p00533_1xkk_fmm_p2_model4/reference/p2_model4_rmsd_replica_04_nm.xvg \
  --xvg examples/egfr_p00533_1xkk_fmm_p2_model4/reference/p2_model4_rmsd_replica_05_nm.xvg \
  --input-unit nm \
  --output-unit angstrom
```

Pass all five files to reproduce the pose-level reference value.

## Stored analysis versus a new run

The example checks the analysis definition, the nm-to-angstrom conversion, and
the maximum-over-time-and-replicas aggregation. It does not launch GROMACS or
recreate the stored trajectory, and a new run is not expected to reproduce this
score exactly.

## Included material

- `reference/` holds the five numeric XVGs and the expected score.
- `prepared_input/` is the lightweight prepared fixture and 5-ns template.
- `config.single_replica.yaml` and `config.five_replicas.yaml` are reduced
  configurations for quick functional examples.
- `metadata.yaml`, `zenodo_mapping.yaml`, and `provenance/` record the pose
  identity, the local pose/RMSD crosswalk, and the applied transformations.

The XVGs keep every numeric source line, and the prepared data keep the source
atom numbering and scientific values.

## Provenance and licensing

Project-generated metadata, docking outputs, RMSD series, and MD-score files
are released under CC BY 4.0. No binding-activity value or activity label is
included. See the [data-license notice](../../docs/legal/data_license.md).

Underlying PDB coordinate data were obtained directly from PDB entry `1XKK`
and remain under the PDB archive terms and CC0. Cite that PDB entry and its
primary structure publication. The MIT software license does not apply to
example/reference data, and this project does not claim copyright over the
underlying PDB coordinates.
