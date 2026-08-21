# Screening template

Use this template for a local screen of many prepared protein–ligand poses.
Copy it to a working directory:

```bash
cp -R templates/screening my_shortmd_screen
cd my_shortmd_screen
```

Create this layout:

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

The prepared directories are not supplied. Create each one and populate it with
its own complete GROMACS system: `start.gro`, `topol.top` and local includes,
`index.ndx`, plus `mdp/nvt.mdp`, `mdp/npt.mdp`, and `mdp/production.mdp`.

Edit the common simulation and resource settings in `run.yaml`, and list one
entry per docked pose in `poses.yaml`.

After populating `prepared/`, check the common configuration and the pose
specified in `input`:

```bash
gpu-shortmd inspect --config run.yaml
```

Check every pose in the manifest and write the task plan without starting MD.
List more GPUs as `--gpu-ids 0,1,2` to spread the tasks across them:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0 \
  --dry-run
```

Review the generated plan, then remove `--dry-run` to execute:

```bash
gpu-shortmd run \
  --config run.yaml \
  --manifest poses.yaml \
  --gpu-ids 0
```

Run directories appear below `outputs/`, relative to `run.yaml`. Start with
`pose_summary.csv`, `replica_summary.csv`, and `run_report.md` in the new run
directory.
