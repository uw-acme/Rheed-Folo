"""Explicit run settings and provenance for files produced outside cell outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def parameter(name: str, default: int) -> int:
    value = int(os.getenv(f"RHEED_{name}", str(default)))
    if value < 1:
        raise ValueError(f"RHEED_{name} must be positive")
    return value


def restore_model_snapshot(path: str | Path, *, custom_objects: dict | None = None):
    """Restore a trusted RHEED inference model from captured JSON and weight tensors.

    QKeras 0.9 stores object activations as dictionaries, but its default
    QActivation constructor expects a callable/string. Decode those explicitly.
    Custom training Lambda functions additionally require their notebook context.
    """
    import numpy as np
    import tensorflow as tf
    from qkeras import QActivation, QConv2D, QDense
    from qkeras.quantizers import get_quantizer

    class RestoringQActivation(QActivation):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            if isinstance(config.get("activation"), dict):
                config["activation"] = get_quantizer(config["activation"])
            return cls(**config)

    objects = {"QActivation": RestoringQActivation, "QConv2D": QConv2D, "QDense": QDense,
               **(custom_objects or {})}
    with np.load(path, allow_pickle=False) as data:
        model = tf.keras.models.model_from_json(str(data["architecture_json"]), custom_objects=objects)
        keys = sorted((k for k in data if k.startswith("weight_")), key=lambda k: int(k.split("_")[1]))
        model.set_weights([data[k] for k in keys])
    return model


def capture_directory(tracker, directory: str | Path, *, role: str = "source"):
    """Archive exact generated files with relative paths, sizes, and hashes."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
    archive = tracker.run_dir / f"{directory.name}.zip"
    manifest = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in files:
            relative = path.relative_to(directory).as_posix()
            data = path.read_bytes()
            manifest.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
            out.writestr(relative, data)
        out.writestr("dataerai-file-manifest.json", json.dumps(manifest, indent=2))
    return tracker.capture_file(archive, role=role, metadata={"file_count": len(files)})


def setup(notebook: str):
    from dataerai_notebook import start_dataerai_notebook
    tracker = start_dataerai_notebook(
        Path(__file__).parent / notebook,
        owner_type=os.getenv("DATAERAI_OWNER_TYPE", "auto"),
        owner_id=os.getenv("DATAERAI_OWNER_ID") or None,
        collection_id=os.getenv("DATAERAI_COLLECTION_ID") or None,
        collection_prefix=os.getenv("DATAERAI_COLLECTION_PREFIX") or None,
        collection_postfix=(
            os.getenv("DATAERAI_COLLECTION_POSTFIX")
            or os.getenv("DATAERAI_COLLECTION_SUFFIX")
            or None
        ),
        binary_path=os.getenv("DATAERAI_BINARY") or None,
    )
    context = {"profile": os.getenv("RHEED_PROFILE", "full"),
               "parameters": {k: v for k, v in os.environ.items() if k.startswith("RHEED_")}}
    context["toolchain"] = {}
    for tool, version_arg in [("g++", "--version"), ("ld", "-v")]:
        executable = shutil.which(tool)
        if executable:
            version = subprocess.run([executable, version_arg], capture_output=True, text=True, timeout=10)
            context["toolchain"][tool] = {"path": executable, "resolved_path": str(Path(executable).resolve()),
                "sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                "version": version.stdout + version.stderr, "exit_code": version.returncode}
    tracker.record_cell(source="# Explicit execution settings", assigned_names=["execution_settings"],
                        user_ns={"execution_settings": context})
    for name in ("pyproject.toml", "uv.lock", "requirements-dataerai.txt", "requirements-rheed-run.txt", "rheed_runtime.py", "dataerai_notebook.py", "dataerai_console_api.py"):
        path = Path(__file__).parent / name
        if path.is_file():
            tracker.capture_file(path, role="source", relationship="uses_dependency")
    for asset_id in json.loads(os.getenv("DATAERAI_UPSTREAM_ASSETS", "[]")):
        tracker._link(tracker.run_asset_id, asset_id, "derived_from")
    return tracker
