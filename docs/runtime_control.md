# Runtime control

This page covers optional pruning, controlled stop, resume, and explicit retry
of failed tasks.

## Optional pruning

Pruning stops the unfinished replicas of a pose once its observed ligand RMSD
exceeds a threshold you choose. It is disabled by default:

```yaml
pruning:
  enabled: false
  threshold_angstrom: null
  grace_period_seconds: 30
```

To enable it, supply a positive threshold in angstroms:

```yaml
pruning:
  enabled: true
  threshold_angstrom: 4.0
  grace_period_seconds: 30
```

The value above shows the syntax only. There is no universal threshold; the
choice requires target- and protocol-specific scientific justification.

## Trigger and pose scope

The comparison is strict:

```text
observed RMSD in Å > threshold in Å
```

Equality does not trigger pruning. When one replica exceeds the threshold:

- the software prunes that pose;
- its running siblings stop;
- its pending siblings are skipped; and
- other poses continue.

A pruned pose does not receive a completed MD-score, so `md_score_angstrom`
stays null. The trigger replica, time, and RMSD, together with the observed
maximum RMSD, are recorded separately. The threshold and observed maximum are
diagnostic values, not substitutes for the pose MD-score.

Online and final observations use the same PBC, fitting, ligand-heavy
selection, and unit definition. See
[Method and limitations](method_and_limitations.md).

## Controlled stop

Stop through the CLI rather than signalling processes manually:

```bash
gpu-shortmd stop RUN_DIR
```

The command stops the run in a controlled way: no new work starts, and the run
keeps the information that resume needs.

## Resume

```bash
gpu-shortmd run --resume RUN_DIR
```

Resume verifies the stored configuration, pose manifest, inputs, completed
work, and available checkpoints before continuing.

It preserves resolved seeds, per-pose status, pruning evidence, completed
replicas, and CPU-thread settings. Verified completed stages are reused, and
an incomplete production stage can continue from a valid GROMACS checkpoint.

## If resume refuses to continue

Run `gpu-shortmd stop RUN_DIR`, review the reported issue and the logs, and
confirm the stop before trying resume again. Do not start a second run for the
same directory, and do not bypass the refusal.

## Retry failed tasks

Resume does not requeue failed tasks on its own. Correct the external cause,
then opt in:

```bash
gpu-shortmd run --resume RUN_DIR --retry-failed
```

The retry keeps the recorded failure history. Tasks belonging to a pruned pose
are never requeued.

## Operational checklist

- Do not change frozen configs, manifests, prepared inputs, or dependency
  files before resume.
- Do not copy a run directory while its processes are still writing to it.
- Preserve checkpoints, completion evidence, logs, manifests, and checksums
  together.
- Do not interpret a null value as zero or as a completed MD-score.

See [Troubleshooting](troubleshooting.md) for common refusal and failure cases.
