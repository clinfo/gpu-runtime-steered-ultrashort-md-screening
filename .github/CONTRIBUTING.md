# Contributing

[Project README](../README.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md)

Changes should preserve the explicit scientific calculation, unit contract,
prepared-system boundary, and fail-closed validation behavior.

Before proposing a change:

1. create a Python 3.11 or 3.12 environment and install `.[dev]`;
2. add or update tests for behavior changes;
3. run the relevant tests, Ruff, and mypy;
4. build and inspect the wheel and source distribution;
5. test a clean installation and the CLI; and
6. run `python tools/release/audit_repository.py --root .` and the package
   audit in `tools/release/`.

Contributor environment definitions are in `dev/`. They are development
resources and are not runtime requirements.

Do not commit simulation binaries, generated run directories, credentials,
personal data, private infrastructure details, or external validation
evidence that is not intended for public distribution.

Scientific-definition changes require an explicit rationale, compatibility
assessment, tests, documentation updates, and release-level validation. Do
not silently substitute another fitting, atom-selection, PBC, unit, or score
aggregation definition.
