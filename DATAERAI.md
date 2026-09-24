# Dataerai integration and provenance

This integration records the existing RHEED workflow as a continuous graph:
notebook source → run → ordered cells → arrays, models, figures and files → summary.
The implementation adapts the QICK recorder from jagar2/QIS_SummerSchool_2024
(commit a89e74998e46fe1cc284bc1be15f72da92720016). Each repository carries its
own copy so a notebook does not depend on a checkout of another repository.

## Install and select a destination

Use Python 3.11 in an isolated environment. For Folo/Gaussian:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-rheed-run.txt
dataerai auth login --server https://beta.dataerai.com
# Or: dataerai auth login --device --client-id dataerai-mobile --server https://beta.dataerai.com
export DATAERAI_OWNER_TYPE=project
export DATAERAI_OWNER_ID=<project-uuid>
```

By default, each notebook uses its own collection named
`<repository> · <notebook>`, such as `Rheed-Folo · Generated_Flow`. An exact
name is reused on later runs. Add a prefix or postfix to route another run of
the same notebook into a different collection:

```sh
python run_dataerai.py Generated_Flow.ipynb --profile smoke \
  --collection-prefix "September batch" --collection-postfix "rerun 2"
```

The equivalent environment variables are `DATAERAI_COLLECTION_PREFIX` and
`DATAERAI_COLLECTION_POSTFIX`; `DATAERAI_COLLECTION_SUFFIX` is accepted as an
alias. `DATAERAI_COLLECTION_ID=<collection-uuid>` remains an explicit override
and bypasses automatic notebook collection resolution. The resolved collection
ID, title, routing mode, prefix and postfix are retained in every record.

The destination is sealed at run start. All records use the selected owner and
collection. Credentials remain in the CLI credential store; environment variables
are not indiscriminately serialized. Do not put secrets in notebook source or print
them: exact source, stdout/stderr and rich displays are research evidence and are
retained. Secret-named assigned values are redacted before artifact serialization.

## Execute notebooks

```sh
python run_dataerai.py Generated_Flow.ipynb --profile smoke
python run_dataerai.py Generated_Flow.ipynb --profile full
RHEED_DATA_DIR=/path/to/Data python run_dataerai.py Real_Flow.ipynb --profile full
```

`full` retains the original dataset sizes and 100 training epochs. `smoke` uses
8/4/4 training/validation/test examples, batch size 4 and one epoch, with those
values recorded. Both use the real notebook generator and QKeras architecture;
a smoke run is not evidence of trained-model accuracy. Explicit `RHEED_` settings
for the five parameters override the profile. `--through training` deliberately
ends before HLS conversion and labels the execution **partial**. HLS conversion,
C++ compilation and inference are attempted by default. On macOS, the generated
hls4ml headers require GNU C++ (Apple Clang reports ambiguous `std::complex`
specializations). Install GCC and put its versioned `g++` behind a `g++` alias in
a run-local directory on PATH; use the system linker rather than an older Conda
linker. Compiler/linker paths, binary hashes and versions are recorded. Synthesis remains the
original notebook's commented-out, user-controlled build step.

The real workflows require the original experimental files:

- Folo: `STO_STO_test6_06292022-standard-compressed.h5`
- Gaussian: `RHEED_4848_test6.h5`

These files are not distributed in the repositories. Missing files produce a
failed recorded execution; generated samples are never substituted for real data.
The full input HDF5 file is copied and uploaded with its hash before reading it.
Use sufficient disk space for the local spool and datasets.

## What is retained

- Full source notebook, including markdown and notebook/cell metadata; exact code,
  source hash, order, timestamps, duration, status and traceback for every executed cell.
- Live stdout/stderr, rich output payloads, image displays and figures.
- NumPy arrays, dictionaries of arrays and TensorFlow tensors as NPZ with shapes and dtypes.
- Keras/QKeras architecture JSON and all numerical model weights, including updated
  weights after training; training history, configuration and assigned scalar values.
- Generated HLS projects and test vectors as ZIP files with per-file SHA-256 manifests.
- Git repository/commit/dirty flag, Python/platform and installed package versions,
  immutable destination, run UUID, settings and dependency hashes.
- Executed `.ipynb`, cell error outputs, terminal summary and server asset IDs.

Restore a saved inference model with:

```python
from rheed_runtime import restore_model_snapshot
model = restore_model_snapshot("baby_yolo_qk.npz")  # or gaussian_qk.npz
predictions = model(images, training=False).numpy()
```

The helper handles the QKeras 0.9 activation-deserialization compatibility issue.
A dedicated CI test round-trips a real quantized model and requires identical
predictions. Load only trusted architecture snapshots. Custom training Lambda
models require their original notebook function/context as well.

The native Dataerai research-object type is set for notebooks, figures, software,
simulation/measurement outputs, analysis, protocols and logs. `generated_by`,
`part_of_run`, `continues_from`, `uses_dependency` and `executes_notebook` relationships
connect the evidence. Set `DATAERAI_UPSTREAM_ASSETS` to a JSON array of real asset
UUIDs to link a run to earlier model/export records across the three repositories.
Do not link a historical bitstream to a newly trained model unless that build actually used it.

Local evidence is kept in `.dataerai/runs/<run-id>/` and
`.dataerai/executions/<execution-id>/`. `asset-index.json` maps serialized files to
uploaded assets. `receipt.json` identifies the run and summary. These directories
are ignored by Git. Capture failures stop execution; temporary connection, gateway
and local database-lock errors have bounded retries. Retrying a complete notebook
creates a new run rather than overwriting the old graph. The recorder verifies that
the uploaded content version is available on the server, completing a pending
byte transfer through the normal API if necessary. Setup and finalization errors
retain an executed notebook locally and make the run fail explicitly.

The runner captures a notebook snapshot before finalization, then saves an additional
local snapshot containing finalization output. Normal interactive Jupyter execution
uses the setup and finalization cells, but use the runner when a durable executed
notebook (including outputs on failure) is required. Unknown arbitrary Python objects
are represented by type/repr; use `dataerai.capture_file(path)` for their lossless
application-native serialization. Optimizer state is not included in the weight NPZ;
use `model.save()` and capture that directory for exact training resumption.

## Hardware pipeline

In Rheed-HW, install `requirements-dataerai.txt` plus `numpy nbclient ipykernel pytest`
and Icarus Verilog (`brew install icarus-verilog` or `apt-get install iverilog`). Run:

```sh
python run_dataerai.py Hardware_Provenance.ipynb
python run_hardware.py --stage synthesis --output 07_vivado_project -- vivado -mode batch -source 03_scripts/create_vivado_project.tcl
```

The hardware notebook performs actual simulation of the existing `nms_top5` RTL
with independent expected peaks, records a VCD waveform and inventories the tracked
source tree. It also retains the existing release bitstream as an explicitly
pre-existing input. This does not establish a new FPGA build, board programming,
full mixed-language Coaxlink simulation or physical camera acquisition. Those need
the lab's Vivado/Vitis installation, licensed Euresys IP and connected hardware.

`run_hardware.py` wraps a supplied simulation/synthesis/implementation/acquisition
command, recording exact arguments, status, streams, elapsed time, selected input
files and output files/directories. Specify `--input` and `--output` repeatedly.
Source-controlled hardware is never reprogrammed merely by importing the integration.

## Verification and review

```sh
python -m pytest tests -q
```

CI exercises capture/failure/destination contracts and notebook coverage. The notebook
repositories also characterize reproducible upstream generated data without network
access. Hardware CI compiles and runs the original RTL testbench. Live Dataerai and
training/HLS evidence is separate from unit-test evidence and is reported in the PR.
No existing scientific algorithms, RTL, firmware or upstream dependency lockfile are
changed. Remove the additive setup/wrappers to roll back; stored runs remain inspectable.

## Recorded validation

See [dataerai-validation.json](docs/dataerai-validation.json) for the saved run IDs,
source commits, CI result and downloaded-artifact hashes from 2026-09-16.
Download saved artifacts through the Dataerai Console. On consolidated storage,
the pinned beta SDK may use an obsolete object prefix; the Console and authorized
`files[].download_url` returned by transfer initiation resolve the stored bytes.
