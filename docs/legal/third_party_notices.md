# Third-party notices

[Project README](../../README.md) · [Software license](../../LICENSE) · [Data license](data_license.md)

External software components listed below are user-supplied or Python runtime
dependencies. The bundled example also uses underlying coordinate data from
the Protein Data Bank.

| Component | Role | License or terms |
|---|---|---|
| GROMACS | Stable external MD and RMSD runtime | LGPL-2.1-or-later |
| NVIDIA driver and CUDA runtime | GPU execution environment | NVIDIA license terms |
| Python | Package runtime | PSF License |
| NumPy | Numeric runtime | BSD-3-Clause |
| packaging | Version parsing | Apache-2.0 or BSD-2-Clause |
| psutil | Process ownership and environment inspection | BSD-3-Clause |
| Pydantic | Strict configuration models | MIT |
| PyYAML | Safe YAML loading | MIT |
| Rich | Console formatting dependency | MIT |
| Typer | CLI framework | MIT |
| Protein Data Bank | Source of underlying PDB entry `1XKK` coordinates | PDB archive terms / CC0 |

Transitive Python dependencies remain under their upstream terms. The package
metadata declares dependencies; no dependency source tree is vendored here.
Users should cite PDB entry `1XKK` and its primary structure publication. No
project copyright is claimed over the underlying PDB coordinates.
