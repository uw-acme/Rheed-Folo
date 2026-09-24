"""Execute an instrumented notebook and persist the executed notebook, even on failure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--through", choices=["training", "hls"], default="hls")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--collection-prefix")
    parser.add_argument(
        "--collection-postfix", "--collection-suffix", dest="collection_postfix"
    )
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    os.environ["RHEED_PROFILE"] = args.profile
    os.environ["RHEED_THROUGH"] = args.through
    if args.collection_prefix is not None:
        os.environ["DATAERAI_COLLECTION_PREFIX"] = args.collection_prefix
    if args.collection_postfix is not None:
        os.environ["DATAERAI_COLLECTION_POSTFIX"] = args.collection_postfix
    if args.profile == "smoke":
        for key, value in {"TRAINING_DATASET_SIZE": 8, "VALIDATION_DATASET_SIZE": 4,
                           "TEST_DATASET_SIZE": 4, "BATCH_SIZE": 4, "NUM_EPOCHS": 1}.items():
            os.environ.setdefault(f"RHEED_{key}", str(value))
    output_dir = notebook_path.parent / ".dataerai" / "executions" / str(uuid.uuid4())
    output_dir.mkdir(parents=True)
    output = output_dir / notebook_path.name
    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(args.workdir.resolve() if args.workdir else notebook_path.parent)}})
    failure = None
    planned_stop = False
    try:
        with client.setup_kernel():
            try:
                for index, cell in enumerate(nb.cells):
                    if "dataerai-finish" in cell.metadata.get("tags", []):
                        continue
                    if args.through == "training" and "# Imports for HLS #" in cell.source:
                        planned_stop = True
                        break
                    print(f"Executing cell {index}: {cell.source[:65]!r}", flush=True)
                    client.execute_cell(cell, index)
            except BaseException as exc:
                failure = exc
            finally:
                nb.metadata["dataerai_execution"] = {"profile": args.profile, "through": args.through,
                    "status": "failed" if failure else "partial" if planned_stop else "completed",
                    "unexecuted_code_cells": [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.execution_count is None]}
                nbformat.write(nb, output)
                cleanup = nbformat.v4.new_code_cell(
                    "from pathlib import Path\nimport json\n"
                    "_tracker = globals().get('_dataerai_active_tracker') or globals().get('dataerai')\n"
                    "if _tracker is not None:\n"
                    f"    _status = {nb.metadata['dataerai_execution']['status']!r}\n"
                    "    try:\n"
                    f"        _tracker.capture_file({str(output)!r}, role='executed_notebook')\n"
                    "    except BaseException:\n"
                    "        _status = 'failed'\n"
                    "        raise\n"
                    "    finally:\n"
                    "        try:\n"
                    "            _tracker.finish(status=_status)\n"
                    "        finally:\n"
                    f"            Path({str(output_dir / 'receipt.json')!r}).write_text(json.dumps({{'run_id': _tracker.run_id, 'run_asset_id': _tracker.run_asset_id, 'summary_asset_id': _tracker.summary_asset_id, 'run_dir': str(_tracker.run_dir), 'cell_count': _tracker.cell_count, 'asset_count': _tracker.asset_count, 'relationship_count': _tracker.relationship_count}}))\n"
                )
                nb.cells.append(cleanup)
                try:
                    client.execute_cell(cleanup, len(nb.cells) - 1)
                except BaseException as exc:
                    print(f"Finalization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    failure = failure or exc
                nbformat.write(nb, output_dir / "execution-with-finalization.ipynb")
    finally:
        nbformat.write(nb, output_dir / "execution-with-finalization.ipynb")
        print(f"Local execution evidence: {output_dir}", flush=True)
    if failure:
        raise failure


if __name__ == "__main__":
    main()
