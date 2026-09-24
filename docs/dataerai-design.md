# Decision: durable RHEED execution provenance

## Problem and scope
Scientific notebook outputs and FPGA build evidence currently stay in local files.
Add opt-in, authenticated provenance using the QICK demo contract, with one PR per
independent upstream repository. Preserve mathematical formulas, model topology,
RTL, firmware, default sample counts and epoch counts. Parameterize counts only
through explicit recorded settings and fix missing output-directory creation.

## Decision and role in the software
The local recorder wraps each scientific code cell with an IPython magic. It stores
content before upload, seals the destination, uses unique run identities, assigns
native record types and connects cell/output/dependency records. The runner owns
kernel execution and failure finalization. File archives supplement assigned-value
capture for HLS and hardware outputs. Metadata records the actual execution scope.
No remote credentials are committed. This is a client integration; no server
schema, permission or API changes are proposed.

## Risks and recovery
Upload errors stop the workflow, retaining local evidence. Large original datasets
can require substantial spool space. Training/compile durations are unchanged for
full runs. Numerical arrays and model parameters are preserved, but arbitrary
objects need explicit native export; documented limitations are not silently treated
as full captures. A failed/partial execution cannot establish physical validation.
Code review and maintainer approval remain required before merge.

## Requirement and invariant verification
| Requirement | Evidence |
|---|---|
| All scientific cells tracked | notebook coverage test; full source notebook artifact |
| Correct destination and separate runs | sealed destination and distinct identity tests |
| One collection per notebook | automatic exact-name collection routing test |
| Same notebook can use another collection | prefix/postfix and runner CLI contract tests |
| Lossless arrays and model weights | NPZ roundtrip and callable-model tests |
| Error remains error | failed-cell, capture-error and real-kernel cleanup tests |
| Source/code/streams preserved | cell-execution JSON and executed notebook |
| Files and generated source traceable | SHA-256 file and archive roundtrip tests |
| Existing algorithms unchanged | upstream generated-data characterization; RTL fixture tests |
| Real-data dependencies honest | no generated replacement for missing HDF5 data |
| Hardware claims scoped | tool exit status, source manifest, waveform, stage labels |

## Validation
Run the new tests, execute generated notebooks with the explicit smoke profile,
attempt the real notebooks against available input files, and run the hardware RTL
simulation. Record unavailable lab data/tools as blockers in the execution evidence
and PR. CI never requires Dataerai credentials. Live run results are documented
separately from algorithm accuracy or device deployment claims.
