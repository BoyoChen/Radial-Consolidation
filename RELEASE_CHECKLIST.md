# Release checklist

## Prepared locally

- [x] Independent Git repository, no dependency on the private research checkout.
- [x] One build command for four figures, three tables and Algorithm 1.
- [x] Bundled numeric inputs, schemas, checksums and regression references.
- [x] Fresh-environment Windows reconstruction and nine unit tests.
- [x] Independent Git clone builds from an external working directory with
  Python network connections blocked; Git checkout preserves data checksums.
- [x] Original manuscript numerical cross-check and rendered PDF review.
- [x] No research orchestration, credentials, internal run names or server paths.
- [x] Portable fonts and pinned direct/transitive dependencies.
- [x] Explicit boundary between figure reproduction and training reproduction.
- [x] Original approved manuscript remains unmodified.

## Before publishing

- [x] Author created the destination repository and configured its remote.
- [x] Author selected MIT for code; the reference repository's DOI is not reused.
- [ ] Author selects a separate research-data/manuscript reuse license if desired.
- [ ] Author checks release timing, coauthor approval and venue anonymity rules.
- [x] Add the verified repository URL and citation metadata; do not imply acceptance.
- [ ] Run the included Windows/Linux GitHub Actions matrix after upload.

The author's existing remote is preserved. No push or public release is performed
by this local integration. The GitHub workflow configuration is supplied, but remote CI is not yet verified.
Do not describe this analysis package as a complete from-scratch training release.
