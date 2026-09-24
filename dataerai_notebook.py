"""Continuous Dataerai provenance capture for the RHEED notebooks.

The public entry point is :func:`start_dataerai_notebook`.  It registers the
``%%dataerai`` IPython cell magic used by every executable tutorial cell.  The
magic preserves the normal notebook display while recording:

* the exact cell source and execution status;
* stdout, stderr, tracebacks, and rich display payloads;
* assigned arrays, tables, figures, configuration dictionaries, and model
  weights and architecture; and
* a continuous chain of Dataerai relationships from the notebook source,
  through the run and each cell, to every captured output.

All files are written to a gitignored local spool before upload.  The tutorial
defaults to ``fail_fast`` so a cell cannot silently continue without its
Dataerai record.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "org.dataerai.rheed-notebook-provenance/v1"
COURSE = {
    "name": "RHEED",
    "repository": "https://github.com/uw-acme",
    "instrument": "RHEED",
}
_SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|cookie|credential)",
    re.IGNORECASE,
)
_SAFE_TAG = re.compile(r"[^a-zA-Z0-9_.:-]+")
_DEPENDENCY_SUFFIXES = {
    ".py",
    ".pkl",
    ".json",
    ".csv",
    ".npy",
    ".npz",
    ".bit",
    ".hwh",
    ".yaml",
    ".yml",
}
_MAX_DEPENDENCY_UPLOAD_BYTES = 2_000_000
_TITLE_MAX_LENGTH = 255
_MANAGED_PROJECT_NAME = "RHEED notebook records"
_FIGURE_FORMATS = {"png", "jpg", "jpeg", "svg"}
_TRANSIENT_DATAERAI_ERROR = re.compile(
    r"HTTP (?:429|500|502|503|504)\b|An internal error occurred\. Please retry the transfer\.|"
    r"timed out|did not complete within|connection (?:was )?closed|database is locked|SQLITE_BUSY",
    re.IGNORECASE,
)
_DEFAULT_TRANSFER_TIMEOUT_SECONDS = 90.0
_DEFAULT_DAEMON_RECYCLE_UPLOADS = 16


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_transient(operation: Any, *, attempts: int = 5) -> Any:
    """Retry short-lived gateway/rate-limit failures without masking real errors."""

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not _TRANSIENT_DATAERAI_ERROR.search(str(exc)):
                raise
            delay = 2 ** attempt
            print(
                f"Dataerai request hit a transient error; retrying in {delay}s "
                f"({attempt}/{attempts - 1}): {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slug(value: str, *, limit: int = 80) -> str:
    cleaned = _SAFE_TAG.sub("-", value.strip()).strip("-").lower()
    return (cleaned or "record")[:limit]


def _title(*parts: str) -> str:
    """Build a readable, collision-resistant title within the API limit."""

    rendered = " · ".join(str(part).strip() for part in parts if str(part).strip())
    if len(rendered) <= _TITLE_MAX_LENGTH:
        return rendered
    digest = _sha256_bytes(rendered.encode("utf-8"))[:12]
    prefix_length = _TITLE_MAX_LENGTH - len(digest) - 3
    return f"{rendered[:prefix_length].rstrip()} · {digest}"


def _uuid_text(value: Any, *, field_name: str) -> str:
    """Return a canonical UUID string or raise an actionable configuration error."""

    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a Dataerai UUID, not an email or display name; "
            f"received {value!r}."
        ) from exc


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _safe_repr(value: Any, *, limit: int = 20_000) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive for vendor objects
        rendered = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(rendered) > limit:
        return rendered[:limit] + f"… <{len(rendered) - limit} characters omitted>"
    return rendered


def _json_safe(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Convert scientific Python values to bounded, secret-aware JSON data."""

    if key and _SECRET_KEY.search(key):
        return "<redacted>"
    if depth > 12:
        return {"python_type": type(value).__qualname__, "repr": _safe_repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "size_bytes": len(value),
            "sha256": _sha256_bytes(value),
        }

    module = type(value).__module__
    if module.startswith("numpy"):
        try:
            if getattr(value, "ndim", 1) == 0:
                return _json_safe(value.item(), depth=depth + 1)
            size = int(value.size)
            summary = {
                "python_type": f"{module}.{type(value).__qualname__}",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "size": size,
            }
            if size <= 10_000:
                summary["values"] = _json_safe(value.tolist(), depth=depth + 1)
            return summary
        except Exception:
            pass

    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= 2_000:
                items["<truncated>"] = f"{len(value) - index} more entries"
                break
            string_key = str(item_key)
            items[string_key] = _json_safe(item_value, key=string_key, depth=depth + 1)
        return items
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        result = [_json_safe(item, depth=depth + 1) for item in sequence[:10_000]]
        if len(sequence) > 10_000:
            result.append({"truncated_items": len(sequence) - 10_000})
        return result

    for method_name in ("model_dump", "to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _json_safe(method(), depth=depth + 1)
            except Exception:
                continue

    return {
        "python_type": f"{module}.{type(value).__qualname__}",
        "repr": _safe_repr(value),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


@dataclass
class DisplayOutput:
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    transient: dict[str, Any] = field(default_factory=dict)
    update: bool = False


@dataclass
class PendingArtifact:
    path: Path
    name: str
    role: str
    file_format: str
    python_type: str
    details: dict[str, Any] = field(default_factory=dict)
    asset_id: str | None = None


class _Tee(io.TextIOBase):
    """Write to the live notebook stream and retain an exact text copy."""

    def __init__(self, live: Any) -> None:
        self.live = live
        self.buffer = io.StringIO()

    def write(self, value: str) -> int:
        self.buffer.write(value)
        return self.live.write(value)

    def flush(self) -> None:
        self.buffer.flush()
        self.live.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.live, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.live, "encoding", "utf-8")

    def getvalue(self) -> str:
        return self.buffer.getvalue()


class NotebookProvenance:
    """Capture one notebook execution as a continuous Dataerai graph."""

    def __init__(
        self,
        notebook_path: str | Path,
        *,
        owner_type: str = "auto",
        owner_id: str | None = None,
        collection_id: str | None = None,
        collection_prefix: str | None = None,
        collection_postfix: str | None = None,
        binary_path: str | None = None,
        on_error: str = "fail_fast",
        spool_root: str | Path | None = None,
        client: Any | None = None,
        shell: Any | None = None,
        console_api: Any | None = None,
    ) -> None:
        if owner_type not in {"auto", "user", "project"}:
            raise ValueError("owner_type must be 'auto', 'user', or 'project'")
        if on_error not in {"fail_fast", "warn"}:
            raise ValueError("on_error must be 'fail_fast' or 'warn'")

        self.notebook_path = Path(notebook_path).expanduser().resolve()
        self.repo_root = self._find_repo_root(self.notebook_path.parent)
        self.owner_type = owner_type
        self.requested_owner_type = owner_type
        self.owner_id = owner_id
        self.owner_resolution: str | None = None
        self.collection_id = collection_id
        env_postfix = os.getenv("DATAERAI_COLLECTION_POSTFIX")
        env_suffix = os.getenv("DATAERAI_COLLECTION_SUFFIX")
        if env_postfix and env_suffix and env_postfix != env_suffix:
            raise ValueError(
                "DATAERAI_COLLECTION_POSTFIX and DATAERAI_COLLECTION_SUFFIX conflict"
            )
        self.collection_prefix = self._validate_collection_affix(
            collection_prefix
            if collection_prefix is not None
            else os.getenv("DATAERAI_COLLECTION_PREFIX"),
            field_name="collection prefix",
        )
        self.collection_postfix = self._validate_collection_affix(
            collection_postfix
            if collection_postfix is not None
            else env_postfix or env_suffix,
            field_name="collection postfix",
        )
        self.collection_title: str | None = None
        self.collection_resolution: str | None = None
        self._upload_destination: dict[str, Any] | None = None
        self._console_api: Any | None = console_api
        self.binary_path = binary_path
        self.on_error = on_error
        self.client = client
        self.shell = shell
        self._owns_client = client is None
        self.run_id = str(uuid.uuid4())
        self._client_socket_path = f"/tmp/dataerai-notebook-{self.run_id}.sock"
        self._uploads_since_client_start = 0
        root = Path(spool_root) if spool_root else self.repo_root / ".dataerai" / "runs"
        self.run_dir = root.expanduser().resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = _utc_now()
        self.finished_at: str | None = None
        self.notebook_asset_id: str | None = None
        self.run_asset_id: str | None = None
        self.dependency_manifest_asset_id: str | None = None
        self.last_cell_asset_id: str | None = None
        self.summary_asset_id: str | None = None
        self.cell_count = 0
        self.asset_count = 0
        self.relationship_count = 0
        self.errors: list[dict[str, Any]] = []
        self.running = False
        self._executing_cell = False
        self._pending_file_assets = []
        self.execution_errors = []
        self.asset_index = []
        self._run_metadata: dict[str, Any] = {}
        self._latest_role_asset: dict[str, str] = {}

    @staticmethod
    def _validate_collection_affix(value: str | None, *, field_name: str) -> str:
        if value is None:
            return ""
        cleaned = " ".join(value.split())
        if not cleaned or len(cleaned) > 80 or any(
            character in value for character in ("/", "\\", "\x00")
        ):
            raise ValueError(
                f"{field_name} must contain 1-80 printable characters without slashes"
            )
        return cleaned

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
        return start

    @property
    def notebook_relative_path(self) -> str:
        try:
            return self.notebook_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return self.notebook_path.name

    def _tutorial_metadata(self) -> dict[str, Any]:
        relative = self.notebook_relative_path
        return {
            "notebook_path": relative,
            "notebook_name": self.notebook_path.stem,
            "repository": self.repo_root.name,
            "track": "generated" if "Generated" in relative else "experimental",
            "topic": "RHEED spot localization, Gaussian fitting, and FPGA pipeline",
            "execution_profile": os.getenv("RHEED_PROFILE", "full"),
            "data_mode": os.getenv("RHEED_DATA_MODE", "generated" if "Generated" in relative else "experimental"),
        }

    def _environment_metadata(self) -> dict[str, Any]:
        commit = _run_git(self.repo_root, "rev-parse", "HEAD")
        status = _run_git(self.repo_root, "status", "--porcelain")
        remote = _run_git(self.repo_root, "remote", "get-url", "origin")
        return {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "kernel": getattr(self.shell, "kernel", None).__class__.__qualname__
            if getattr(self.shell, "kernel", None) is not None
            else type(self.shell).__qualname__,
            "packages": {
                "dataerai-sdk": _package_version("dataerai-sdk"),
                "ipython": _package_version("ipython"),
                "jupyterlab": _package_version("jupyterlab"),
                "numpy": _package_version("numpy"),
                "matplotlib": _package_version("matplotlib"),
                **{d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            },
            "git": {
                "repository": remote or COURSE["repository"],
                "commit": commit,
                "dirty_at_start": bool(status),
            },
        }

    def _base_metadata(self, record_kind: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "record_kind": record_kind,
            "course": COURSE,
            "tutorial": self._tutorial_metadata(),
            "destination": self._destination_metadata(),
            "run": {
                "run_id": self.run_id,
                "started_at": self.started_at,
            },
        }

    def _origin_metadata(self, record_kind: str) -> dict[str, Any]:
        """Metadata for reusable, content-addressed source/dependency assets."""

        return {
            "schema": SCHEMA,
            "record_kind": record_kind,
            "course": COURSE,
            "tutorial": self._tutorial_metadata(),
            "destination": self._destination_metadata(),
        }

    def _destination_metadata(self) -> dict[str, str | None]:
        """Return the immutable owner and collection selected when the run began."""

        if self._upload_destination is None:
            raise RuntimeError("The notebook upload destination has not been resolved")
        return dict(self._upload_destination)

    def _tags(self, record_kind: str, *extra: str) -> list[str]:
        return [
            "data-notebook",
            "rheed",
            "rheed-pipeline",
            "jupyter",
            "continuous-provenance",
            f"rheed-run:{self.run_id}",
            f"record-kind:{_slug(record_kind)}",
            *[_slug(item) for item in extra if item],
        ]

    def _origin_tags(self, record_kind: str, *extra: str) -> list[str]:
        return [
            "data-notebook",
            "rheed",
            "rheed-pipeline",
            "jupyter",
            "continuous-provenance",
            f"record-kind:{_slug(record_kind)}",
            *[_slug(item) for item in extra if item],
        ]

    @property
    def _owner_state_path(self) -> Path:
        return self.repo_root / ".dataerai" / "owner-projects.json"

    def _managed_project_id(self, user_email: str) -> str:
        """Create once, then reuse a personal project for legacy beta SDKs."""

        state: dict[str, Any] = {"schema": SCHEMA, "owners": {}}
        if self._owner_state_path.is_file():
            try:
                loaded = json.loads(self._owner_state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("owners"), dict):
                    state = loaded
            except (OSError, json.JSONDecodeError):
                pass

        owner = state["owners"].get(user_email, {})
        existing = owner.get("project_id") if isinstance(owner, dict) else None
        if existing:
            self.owner_resolution = "managed_project_reused"
            return _uuid_text(existing, field_name="managed project_id")

        create_project = getattr(self.client, "create_project", None)
        if not callable(create_project):
            raise RuntimeError(
                "This Dataerai SDK does not expose the signed-in user UUID. Set "
                "DATAERAI_OWNER_TYPE=project and DATAERAI_OWNER_ID=<project-uuid>, "
                "or install a newer provenance-capable SDK."
            )
        project = create_project(
            _MANAGED_PROJECT_NAME,
            description=(
                "Automatically created by the RHEED notebooks so "
                "Dataerai records have a valid UUID owner."
            ),
        )
        project_id = _uuid_text(
            getattr(project, "project_id", None), field_name="created project_id"
        )
        state["schema"] = SCHEMA
        state["owners"][user_email] = {
            "project_id": project_id,
            "project_name": _MANAGED_PROJECT_NAME,
            "created_at": _utc_now(),
        }
        _write_json(self._owner_state_path, state)
        try:
            self._owner_state_path.chmod(0o600)
        except OSError:
            pass
        self.owner_resolution = "managed_project_created"
        return project_id

    def _resolve_owner(self, auth: Any) -> None:
        """Resolve the upload owner without ever substituting an email for a UUID."""

        if self.owner_id:
            if self.owner_type == "auto":
                raise ValueError(
                    "Set DATAERAI_OWNER_TYPE to 'user' or 'project' when "
                    "DATAERAI_OWNER_ID is provided."
                )
            self.owner_id = _uuid_text(
                self.owner_id, field_name="DATAERAI_OWNER_ID"
            )
            self.owner_resolution = "explicit"
            return

        auth_user_id = getattr(auth, "user_id", None)
        if self.owner_type in {"auto", "user"} and auth_user_id:
            self.owner_type = "user"
            self.owner_id = _uuid_text(auth_user_id, field_name="auth_status().user_id")
            self.owner_resolution = "authenticated_user"
            return

        if self.owner_type == "user":
            raise RuntimeError(
                "This SDK reports the signed-in email but not the UUID required for "
                "a user-owned upload. Use DATAERAI_OWNER_TYPE=auto (recommended) "
                "to create/reuse the tutorial project, or set a project UUID explicitly."
            )
        if self.owner_type == "project":
            raise ValueError(
                "DATAERAI_OWNER_ID=<project-uuid> is required when "
                "DATAERAI_OWNER_TYPE=project."
            )

        self.owner_type = "project"
        self.owner_id = self._managed_project_id(auth.user_email)

    def _resolve_notebook_collection(self) -> None:
        """Route each notebook to a stable collection selected by optional affixes."""
        if self.collection_id is not None:
            self.collection_id = _uuid_text(
                self.collection_id,
                field_name="DATAERAI_COLLECTION_ID",
            )
            self.collection_resolution = "explicit"
            return
        if self.owner_id is None:
            raise RuntimeError("The notebook upload owner has not been resolved")
        if self._console_api is None:
            from dataerai_console_api import DataeraiConsoleAPI

            self._console_api = DataeraiConsoleAPI()
        title_parts = [
            self.collection_prefix,
            self.repo_root.name,
            self.notebook_path.stem,
            self.collection_postfix,
        ]
        title = " · ".join(part for part in title_parts if part)
        if len(title) > 255:
            raise ValueError("resolved notebook collection title exceeds 255 characters")
        collection = self._console_api.get_or_create_notebook_collection(
            owner_type=self.owner_type,
            owner_id=self.owner_id,
            title=title,
            description=(
                f"Dataerai provenance records for {self.repo_root.name}/"
                f"{self.notebook_relative_path}."
            ),
            tags=[
                "rheed",
                "notebook-runs",
                f"repository:{_slug(self.repo_root.name).replace('_', '-')}",
                f"notebook:{_slug(self.notebook_path.stem).replace('_', '-')}",
            ],
        )
        self.collection_id = _uuid_text(
            collection.get("id"), field_name="resolved collection id"
        )
        # Personal collection creates may be routed into the user's default project.
        original_owner = (self.owner_type, self.owner_id)
        self.owner_type = str(collection.get("owner_type") or self.owner_type)
        self.owner_id = _uuid_text(
            collection.get("owner_id") or self.owner_id,
            field_name="resolved collection owner id",
        )
        self.collection_title = str(collection.get("title") or title)
        self.collection_resolution = "notebook_collection_created_or_reused"
        if (self.owner_type, self.owner_id) != original_owner:
            self.owner_resolution = "collection_owner_routed"

    def start(self) -> "NotebookProvenance":
        if self.running:
            return self
        if not self.notebook_path.is_file():
            raise FileNotFoundError(f"Notebook does not exist: {self.notebook_path}")

        if self.shell is None:
            try:
                from IPython import get_ipython
            except ImportError as exc:  # pragma: no cover - notebook dependency
                raise RuntimeError("IPython is required for notebook capture") from exc
            self.shell = get_ipython()
        if self.shell is None:
            raise RuntimeError(
                "start_dataerai_notebook() must run inside IPython/Jupyter"
            )

        if self.client is None:
            try:
                from dataerai import DataeraiClient
            except ImportError as exc:
                raise RuntimeError(
                    "Install the integration with: "
                    "python -m pip install -r requirements-dataerai.txt"
                ) from exc
            resolved_binary = (
                self.binary_path
                or os.environ.get("DATAERAI_BINARY")
                or shutil.which("dataerai")
            )
            if not resolved_binary:
                raise RuntimeError(
                    "The dataerai command was not found. Install the pinned CLI from "
                    "requirements-dataerai.txt or set "
                    "DATAERAI_BINARY to the binary path."
                )
            self.binary_path = resolved_binary
            self.client = DataeraiClient(
                socket_path=self._client_socket_path,
                binary_path=resolved_binary,
            )

        self.client.connect()
        if not callable(getattr(self.client, "create_relationship", None)):
            raise RuntimeError(
                "This notebook requires a provenance-capable Dataerai SDK. "
                "Install the pinned beta SDK and CLI with: "
                "python -m pip install -r requirements-dataerai.txt"
            )
        auth = self.client.auth_status()
        self._resolve_owner(auth)
        self._resolve_notebook_collection()
        if self.owner_id is None:  # Defensive: _resolve_owner() must set this.
            raise RuntimeError("The notebook upload owner has not been resolved")
        self._upload_destination = {
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "collection_id": self.collection_id,
            "collection_title": self.collection_title,
            "collection_resolution": self.collection_resolution,
            "collection_prefix": self.collection_prefix,
            "collection_postfix": self.collection_postfix,
        }

        notebook_metadata = self._origin_metadata("notebook_source")
        notebook_sha256 = _sha256_bytes(self.notebook_path.read_bytes())
        notebook_metadata["source"] = {
            "sha256": notebook_sha256,
            "size_bytes": self.notebook_path.stat().st_size,
            "identity": (
                f"{self.notebook_relative_path}@sha256:{notebook_sha256}"
            ),
        }
        self.notebook_asset_id = self._upload(
            self.notebook_path,
            title=_title(
                "RHEED notebook source",
                self.notebook_relative_path,
                f"sha256:{notebook_sha256[:12]}",
            ),
            description=(
                "Versioned Jupyter notebook source for a provenance-tracked "
                "RHEED execution."
            ),
            metadata=notebook_metadata,
            tags=self._origin_tags(
                "notebook_source",
                f"rheed-track:{self._tutorial_metadata()['track']}",
                "asset-type:jupyter-notebook",
            ),
        )

        self._run_metadata = self._base_metadata("notebook_run")
        self._run_metadata["environment"] = self._environment_metadata()
        self._run_metadata["authentication"] = {
            "user_email": auth.user_email,
            "user_id": getattr(auth, "user_id", None),
            "requested_owner_type": self.requested_owner_type,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "owner_resolution": self.owner_resolution,
            "collection_id": self.collection_id,
        }
        run_manifest = self.run_dir / "run-manifest.json"
        _write_json(run_manifest, self._run_metadata)
        self.run_asset_id = self._upload(
            run_manifest,
            title=_title(
                "RHEED notebook run", self.notebook_relative_path, self.run_id
            ),
            description=(
                "Execution manifest anchoring every cell and output in this "
                "continuous notebook provenance chain."
            ),
            metadata=self._run_metadata,
            tags=self._tags("notebook_run", "asset-type:experiment-run"),
        )
        # Expose the run before dependency uploads so runner cleanup can finalize
        # a setup failure even when the caller assignment never completes.
        self.running = True
        self.shell.user_ns["_dataerai_active_tracker"] = self
        self._link(
            self.run_asset_id,
            self.notebook_asset_id,
            "executes_notebook",
            qualifiers={"run_id": self.run_id},
        )
        self._capture_dependencies()

        self.shell.register_magic_function(
            self._cell_magic, magic_kind="cell", magic_name="dataerai"
        )
        self.running = True
        print(
            f"Dataerai provenance active for {self.notebook_relative_path}\n"
            f"  signed in: {auth.user_email}\n"
            f"  owner:     {self.owner_type} {self.owner_id} "
            f"({self.owner_resolution})\n"
            f"  run id:    {self.run_id}\n"
            f"  run asset: {self.run_asset_id}\n"
            f"  local spool: {self.run_dir}"
        )
        return self

    def _discover_dependencies(self) -> list[Path]:
        """Find local code, calibration, and firmware referenced by a notebook."""

        dependencies: set[Path] = set()
        module_queue = list(self._notebook_import_modules())
        visited_modules: set[str] = set()
        while module_queue:
            module = module_queue.pop()
            if module in visited_modules:
                continue
            visited_modules.add(module)
            module_path = self._resolve_local_module(module)
            if module_path is None:
                continue
            dependencies.add(module_path)
            try:
                tree = ast.parse(module_path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            module_queue.extend(self._imports_from_tree(tree))

        for reference in self._notebook_file_references():
            resolved = self._resolve_dependency_reference(reference)
            if resolved is not None:
                dependencies.add(resolved)
                if resolved.suffix.lower() == ".bit":
                    hardware_description = resolved.with_suffix(".hwh")
                    if hardware_description.is_file():
                        dependencies.add(hardware_description.resolve())

        local: list[Path] = []
        for dependency in dependencies:
            resolved = dependency.resolve()
            if resolved == self.notebook_path:
                continue
            try:
                resolved.relative_to(self.repo_root)
            except ValueError:
                continue
            local.append(resolved)
        return sorted(set(local))

    def _notebook_code_trees(self) -> list[ast.AST]:
        try:
            notebook = json.loads(self.notebook_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        trees: list[ast.AST] = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            raw_source = cell.get("source", "")
            source = (
                "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)
            )
            if source.startswith("%%dataerai\n"):
                source = source[len("%%dataerai\n") :]
            try:
                transformed = self.shell.transform_cell(source)
                trees.append(ast.parse(transformed))
            except Exception:
                continue
        return trees

    @staticmethod
    def _imports_from_tree(tree: ast.AST) -> list[str]:
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        return modules

    def _notebook_import_modules(self) -> list[str]:
        return sorted(
            {
                module
                for tree in self._notebook_code_trees()
                for module in self._imports_from_tree(tree)
            }
        )

    def _resolve_local_module(self, module: str) -> Path | None:
        relative = Path(*module.split("."))
        for base in (self.notebook_path.parent, self.repo_root):
            for candidate in (
                base / relative.with_suffix(".py"),
                base / relative / "__init__.py",
            ):
                resolved = candidate.resolve()
                if resolved.is_file():
                    try:
                        resolved.relative_to(self.repo_root)
                    except ValueError:
                        continue
                    return resolved
        return None

    def _notebook_file_references(self) -> list[str]:
        """Extract file-like string literals from executable cells, not comments."""

        references: set[str] = set()
        for tree in self._notebook_code_trees():
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    candidate = node.value.strip()
                    if (
                        "\n" not in candidate
                        and Path(candidate).suffix.lower() in _DEPENDENCY_SUFFIXES
                    ):
                        references.add(candidate)
        return sorted(references)

    def _resolve_dependency_reference(self, reference: str) -> Path | None:
        candidate = Path(reference)
        search = (
            [candidate]
            if candidate.is_absolute()
            else [self.notebook_path.parent / candidate, self.repo_root / candidate]
        )
        for path in search:
            resolved = path.expanduser().resolve()
            if resolved.is_file():
                return resolved
        # Some legacy notebooks were moved without updating their relative
        # firmware prefix. Recover only an unambiguous repository match whose
        # trailing directory and filename still agree with the recorded path.
        cleaned_parts = [
            part for part in candidate.parts if part not in {"/", "..", "."}
        ]
        if len(cleaned_parts) >= 2:
            trailing = tuple(cleaned_parts[-2:])
            matches = [
                path.resolve()
                for path in self.repo_root.rglob(candidate.name)
                if tuple(path.parts[-2:]) == trailing
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    def _capture_dependencies(self) -> None:
        dependencies = self._discover_dependencies()
        file_references = self._notebook_file_references()
        if not dependencies and not file_references:
            return
        git_commit = (
            self._run_metadata.get("environment", {}).get("git", {}).get("commit")
        )
        manifest_entries = []
        for path in dependencies:
            relative = path.relative_to(self.repo_root).as_posix()
            size_bytes = path.stat().st_size
            manifest_entries.append(
                {
                    "path": relative,
                    "sha256": _sha256_bytes(path.read_bytes()),
                    "size_bytes": size_bytes,
                    "git_commit": git_commit,
                    "content_uploaded": size_bytes <= _MAX_DEPENDENCY_UPLOAD_BYTES,
                }
            )

        unresolved_references = [
            {
                "reference": reference,
                "status": "not_found_at_capture",
                "note": (
                    "The notebook records this path, but the file was not available "
                    "under the notebook directory or repository root when the run began."
                ),
            }
            for reference in file_references
            if self._resolve_dependency_reference(reference) is None
        ]
        manifest_payload = {
            **self._base_metadata("dependency_manifest"),
            "dependencies": manifest_entries,
            "unresolved_references": unresolved_references,
            "policy": {
                "small_file_upload_limit_bytes": _MAX_DEPENDENCY_UPLOAD_BYTES,
                "large_files": (
                    "Checksum- and Git-pinned in this manifest instead of being "
                    "re-uploaded for every notebook run."
                ),
            },
        }
        manifest_path = self.run_dir / "dependency-manifest.json"
        _write_json(manifest_path, manifest_payload)
        self.dependency_manifest_asset_id = self._upload(
            manifest_path,
            title=_title(
                "RHEED dependency manifest", self.notebook_relative_path, self.run_id
            ),
            description=(
                "Checksums, sizes, repository commit, and capture policy for all "
                "local code, calibration data, and firmware used by this notebook."
            ),
            metadata=manifest_payload,
            tags=self._tags("dependency_manifest", "asset-type:software-environment"),
        )
        self._link(
            self.run_asset_id,
            self.dependency_manifest_asset_id,
            "uses_dependency_manifest",
            qualifiers={
                "dependency_count": len(dependencies),
                "unresolved_reference_count": len(unresolved_references),
            },
        )

        for path, entry in zip(dependencies, manifest_entries):
            if not entry["content_uploaded"]:
                continue
            suffix = path.suffix.lower()
            role = "software_source" if suffix == ".py" else "input_data"
            metadata = {
                "schema": SCHEMA,
                "record_kind": "notebook_dependency",
                "course": COURSE,
                "dependency": {**entry, "role": role},
            }
            dependency_asset_id = self._upload(
                path,
                title=_title(
                    "RHEED notebook dependency",
                    entry["path"],
                    f"sha256:{entry['sha256'][:12]}",
                ),
                description=(
                    f"Content-addressed {role.replace('_', ' ')} used by RHEED "
                    "pipeline notebook runs."
                ),
                metadata=metadata,
                tags=self._origin_tags(
                    "notebook_dependency",
                    f"dependency-role:{role}",
                    f"format:{suffix.lstrip('.') or 'binary'}",
                ),
            )
            self._link(
                self.run_asset_id,
                dependency_asset_id,
                "uses_dependency",
                qualifiers={
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "role": role,
                },
            )
            self._link(
                self.dependency_manifest_asset_id,
                dependency_asset_id,
                "describes_dependency",
                qualifiers={"path": entry["path"]},
            )

    def _upload(
        self,
        path: Path,
        *,
        title: str,
        description: str,
        metadata: dict[str, Any],
        tags: list[str],
    ) -> str:
        recycle_after = int(
            os.environ.get(
                "DATAERAI_DAEMON_RECYCLE_UPLOADS",
                str(_DEFAULT_DAEMON_RECYCLE_UPLOADS),
            )
        )
        if (
            self._owns_client
            and recycle_after > 0
            and self._uploads_since_client_start >= recycle_after
        ):
            self._recycle_owned_client()

        destination = self._destination_metadata()
        record_type = self._record_type_for_upload(path, metadata)
        routed_metadata = {
            **metadata,
            "destination": destination,
            "record_type": record_type,
        }
        transfer_timeout_s = float(
            os.environ.get(
                "DATAERAI_TRANSFER_TIMEOUT_S",
                str(_DEFAULT_TRANSFER_TIMEOUT_SECONDS),
            )
        )

        def upload_once() -> Any:
            try:
                return self.client.upload(
                    str(path),
                    title=title,
                    description=description,
                    owner_type=destination["owner_type"],
                    owner_id=destination["owner_id"],
                    collection_id=destination["collection_id"],
                    tags=tags,
                    metadata=_json_safe(routed_metadata),
                    transfer_timeout_s=transfer_timeout_s,
                )
            except Exception as exc:
                if self._owns_client and _TRANSIENT_DATAERAI_ERROR.search(str(exc)):
                    self._recycle_owned_client()
                raise

        result = _retry_transient(upload_once)
        if getattr(result, "transfer_id", None) and getattr(result, "content_id", None):
            if self._console_api is None:
                from dataerai_console_api import DataeraiConsoleAPI
                self._console_api = DataeraiConsoleAPI()
            _retry_transient(lambda: self._console_api.ensure_upload_complete(
                result.asset_id, result.content_id, result.transfer_id))
        _retry_transient(
            lambda: self._set_asset_record_type(result.asset_id, record_type)
        )
        self.asset_count += 1
        self._uploads_since_client_start += 1
        self.asset_index.append({"asset_id": result.asset_id, "path": str(path.relative_to(self.run_dir)) if path.is_relative_to(self.run_dir) else str(path), "metadata": routed_metadata})
        _write_json(self.run_dir / "asset-index.json", self.asset_index)
        return result.asset_id

    def _recycle_owned_client(self) -> None:
        """Replace a long-lived transfer daemon after a transient stall."""

        if not self._owns_client:
            return
        if self.client is not None:
            self.client.close()
        try:
            Path(self._client_socket_path).unlink()
        except FileNotFoundError:
            pass

        from dataerai import DataeraiClient

        self.client = DataeraiClient(
            socket_path=self._client_socket_path,
            binary_path=self.binary_path,
        )
        self.client.connect()
        self._uploads_since_client_start = 0

    def _record_type_for_upload(
        self,
        path: Path,
        metadata: Mapping[str, Any],
    ) -> str:
        """Map each provenance artifact to Dataerai's research-object taxonomy."""

        record_kind = str(metadata.get("record_kind") or "")
        is_simulation = self._tutorial_metadata()["data_mode"] in {"generated", "simulation"}
        run_type = "protocol_workflow" if self._tutorial_metadata()["data_mode"] == "build" else ("simulation" if is_simulation else "measurement")

        if record_kind in {"notebook_source", "executed_notebook"}:
            return "jupyter_notebook"
        if record_kind == "notebook_dependency":
            dependency = metadata.get("dependency") or {}
            if path.suffix.lower() in {".py", ".bit", ".hwh"}:
                return "software_code"
            if dependency.get("role") == "input_data":
                return "calibration"
            return "software_code"
        if record_kind == "dependency_manifest":
            return "protocol_workflow"
        if record_kind == "notebook_run":
            return run_type
        if record_kind in {"cell_execution", "notebook_run_summary"}:
            return "log"
        if record_kind in {"cell_output", "file_artifact"}:
            output = metadata.get("output") or {}
            file_format = str(output.get("format") or path.suffix.lstrip(".")).lower()
            role = str(output.get("role") or "")
            name = str(output.get("name") or path.stem).lower()
            if file_format in _FIGURE_FORMATS:
                return "figure"
            if role in {"model", "firmware", "source"}:
                return "software_code"
            if role == "configuration":
                if re.search(r"(?:^|_)(?:prog|program)(?:$|_)", name):
                    return "software_code"
                return "protocol_workflow"
            if role == "raw_data":
                return run_type
            return "analysis"
        return "analysis"

    def _set_asset_record_type(self, asset_id: str, record_type: str) -> None:
        """Set the real Console field, using the public SDK when it exposes it."""

        setter = getattr(self.client, "set_record_type", None)
        if callable(setter):
            setter(asset_id, record_type=record_type)
            return
        if self._console_api is None:
            from dataerai_console_api import DataeraiConsoleAPI

            self._console_api = DataeraiConsoleAPI()
        self._console_api.set_record_type(asset_id, record_type)

    def _link(
        self,
        from_asset_id: str,
        to_asset_id: str,
        relationship_type: str,
        *,
        qualifiers: dict[str, Any] | None = None,
        qualifier_note: str | None = None,
        analysis_mode: str | None = None,
    ) -> None:
        try:
            _retry_transient(
                lambda: self.client.create_relationship(
                    from_asset_id,
                    to_asset_id,
                    relationship_type=relationship_type,
                    qualifier_time=_utc_now(),
                    qualifier_note=qualifier_note,
                    analysis_mode=analysis_mode,
                    qualifiers=_json_safe(qualifiers or {}),
                )
            )
        except Exception as exc:
            if getattr(exc, "code", None) == "ERR_RELATIONSHIP_EXISTS":
                return
            raise
        self.relationship_count += 1

    def _cell_magic(self, line: str, cell: str) -> None:
        del line
        if not self.running:
            raise RuntimeError("Dataerai provenance is not active for this notebook")

        self._executing_cell = True
        started_at = _utc_now()
        monotonic_start = time.monotonic()
        stdout_tee = _Tee(sys.stdout)
        stderr_tee = _Tee(sys.stderr)
        displays: list[DisplayOutput] = []
        publisher = getattr(self.shell, "display_pub", None)
        original_publish = getattr(publisher, "publish", None)
        original_showtraceback = getattr(self.shell, "showtraceback", None)
        original_showsyntaxerror = getattr(self.shell, "showsyntaxerror", None)

        if original_publish is not None:

            def capture_publish(*args: Any, **kwargs: Any) -> Any:
                data = kwargs.get("data")
                if data is None and args:
                    data = args[0]
                if isinstance(data, dict):
                    displays.append(
                        DisplayOutput(
                            data=dict(data),
                            metadata=dict(kwargs.get("metadata") or {}),
                            transient=dict(kwargs.get("transient") or {}),
                            update=bool(kwargs.get("update", False)),
                        )
                    )
                return original_publish(*args, **kwargs)

            publisher.publish = capture_publish

        original_stdout, original_stderr = sys.stdout, sys.stderr
        result = None
        direct_error: BaseException | None = None
        direct_traceback = None
        try:
            sys.stdout, sys.stderr = stdout_tee, stderr_tee
            # The outer magic re-raises after the failed cell has been recorded.
            # Suppress the nested renderer so Jupyter shows one traceback, not two.
            if original_showtraceback is not None:
                self.shell.showtraceback = lambda *args, **kwargs: None
            if original_showsyntaxerror is not None:
                self.shell.showsyntaxerror = lambda *args, **kwargs: None
            try:
                result = self.shell.run_cell(cell, store_history=False)
            except BaseException as exc:  # preserve interrupts and SystemExit too
                direct_error = exc
                direct_traceback = exc.__traceback__
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            if original_publish is not None:
                publisher.publish = original_publish
            if original_showtraceback is not None:
                self.shell.showtraceback = original_showtraceback
            if original_showsyntaxerror is not None:
                self.shell.showsyntaxerror = original_showsyntaxerror

        # Some display hooks do not call display_pub for the final expression.
        final_result = getattr(result, "result", None)
        if final_result is not None:
            try:
                data, metadata = self.shell.display_formatter.format(final_result)
                if data and not any(item.data == data for item in displays):
                    displays.append(DisplayOutput(data=data, metadata=metadata or {}))
            except Exception:
                pass

        success = direct_error is None and bool(getattr(result, "success", True))
        error = (
            direct_error
            or getattr(result, "error_before_exec", None)
            or getattr(result, "error_in_exec", None)
        )
        try:
            self.record_cell(
                source=cell,
                stdout=stdout_tee.getvalue(),
                stderr=stderr_tee.getvalue(),
                displays=displays,
                success=success,
                error=error,
                started_at=started_at,
                duration_seconds=time.monotonic() - monotonic_start,
                assigned_names=self._assigned_names(cell),
                user_ns=self.shell.user_ns,
            )
        except Exception as exc:
            self._record_capture_error(exc, cell)
            if self.on_error == "fail_fast":
                raise
            warnings.warn(
                f"Dataerai capture failed; local files remain in {self.run_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self._executing_cell = False
        if direct_error is not None:
            raise direct_error.with_traceback(direct_traceback)
        if error is not None:
            raise error

    def _assigned_names(self, source: str) -> list[str]:
        try:
            transformed = self.shell.transform_cell(source)
            tree = ast.parse(transformed)
        except Exception:
            return []
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, ast.Store
            ):
                root = node.value
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name):
                    names.add(root.id)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return sorted(name for name in names if not name.startswith("_"))

    def _record_capture_error(self, exc: Exception, source: str) -> None:
        payload = {
            "at": _utc_now(),
            "cell_sequence": self.cell_count + 1,
            "source_sha256": _sha256_bytes(source.encode("utf-8")),
            "error_type": type(exc).__qualname__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.errors.append(payload)
        _write_json(self.run_dir / "capture-errors.json", self.errors)

    def record_cell(
        self,
        *,
        source: str,
        stdout: str = "",
        stderr: str = "",
        displays: Iterable[DisplayOutput] = (),
        success: bool = True,
        error: BaseException | None = None,
        started_at: str | None = None,
        duration_seconds: float = 0.0,
        assigned_names: Iterable[str] = (),
        user_ns: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist one executed cell and return its Dataerai asset ID.

        This method is public to make the capture logic testable and usable by
        non-IPython runners.  Normal tutorial cells call it through the magic.
        """

        if not self.run_asset_id:
            raise RuntimeError("Call start() before recording cells")
        if not success:
            self.execution_errors.append({"cell_sequence": self.cell_count + 1, "error": str(error)})
        sequence = self.cell_count + 1
        cell_dir = self.run_dir / f"cell-{sequence:04d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256_bytes(source.encode("utf-8"))
        completed_at = _utc_now()

        displays = list(displays)
        _write_json(cell_dir / "execution-local.json", {
            "source": source, "source_sha256": source_hash, "stdout": stdout, "stderr": stderr,
            "status": "succeeded" if success else "failed", "started_at": started_at,
            "completed_at": completed_at, "duration_seconds": duration_seconds,
            "error": str(error) if error is not None else None,
            "displays": [{"data": d.data, "metadata": d.metadata, "transient": d.transient, "update": d.update} for d in displays],
        })
        artifacts: list[PendingArtifact] = []
        rich_output_manifest: list[dict[str, Any]] = []
        for display_index, display in enumerate(displays, start=1):
            display_record: dict[str, Any] = {
                "index": display_index,
                "metadata": _json_safe(display.metadata),
                "transient": _json_safe(display.transient),
                "update": display.update,
                "mime": {},
            }
            for mime, value in display.data.items():
                if mime in {"image/png", "image/jpeg", "image/svg+xml"}:
                    extension = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/svg+xml": "svg",
                    }[mime]
                    output_path = cell_dir / f"display-{display_index:02d}.{extension}"
                    if mime == "image/svg+xml":
                        output_path.write_text(str(value), encoding="utf-8")
                    else:
                        raw = (
                            value
                            if isinstance(value, bytes)
                            else base64.b64decode(value)
                        )
                        output_path.write_bytes(raw)
                    artifacts.append(
                        PendingArtifact(
                            path=output_path,
                            name=f"display-{display_index:02d}",
                            role="analysis",
                            file_format=extension,
                            python_type=mime,
                            details={"mime_type": mime, "display_index": display_index},
                        )
                    )
                    display_record["mime"][mime] = {
                        "artifact": output_path.name,
                        "sha256": _sha256_bytes(output_path.read_bytes()),
                    }
                else:
                    display_record["mime"][mime] = _json_safe(value)
            rich_output_manifest.append(display_record)

        namespace = user_ns or {}
        # Training mutates an existing callable model without assigning it again.
        assigned_names = set(assigned_names) | {
            name for name, value in namespace.items()
            if not name.startswith("_") and not isinstance(value, type) and callable(getattr(value, "get_weights", None))
            and callable(getattr(value, "to_json", None))
        }
        scalar_values: dict[str, Any] = {}
        for name in sorted(set(assigned_names)):
            if name not in namespace:
                continue
            value = namespace[name]
            if _SECRET_KEY.search(name):
                scalar_values[name] = "<redacted>"
                continue
            artifact = self._serialize_assigned_value(cell_dir, name, value)
            if artifact is not None:
                artifacts.append(artifact)
            elif not isinstance(value, types.ModuleType) and not callable(value):
                scalar_values[name] = _json_safe(value, key=name)
        if scalar_values:
            scalar_path = cell_dir / "assigned-values.json"
            _write_json(scalar_path, scalar_values)
            artifacts.append(
                PendingArtifact(
                    path=scalar_path,
                    name="assigned-values",
                    role="derived_data",
                    file_format="json",
                    python_type="mapping",
                    details={"variables": sorted(scalar_values)},
                )
            )

        uploaded_outputs: list[dict[str, Any]] = []
        for artifact in artifacts:
            artifact_metadata = self._base_metadata("cell_output")
            artifact_metadata["run"]["cell_sequence"] = sequence
            artifact_metadata["run"]["cell_source_sha256"] = source_hash
            artifact_metadata["output"] = {
                "name": artifact.name,
                "role": artifact.role,
                "format": artifact.file_format,
                "python_type": artifact.python_type,
                "sha256": _sha256_bytes(artifact.path.read_bytes()),
                "size_bytes": artifact.path.stat().st_size,
                **artifact.details,
            }
            artifact.asset_id = self._upload(
                artifact.path,
                title=_title(
                    "RHEED notebook output",
                    self.notebook_relative_path,
                    self.run_id,
                    f"cell {sequence:04d}",
                    artifact.name,
                ),
                description=(
                    f"{artifact.role.replace('_', ' ').title()} produced by cell "
                    f"{sequence} of notebook run {self.run_id}."
                ),
                metadata=artifact_metadata,
                tags=self._tags(
                    "cell_output",
                    f"cell:{sequence:04d}",
                    f"output-role:{artifact.role}",
                    f"format:{artifact.file_format}",
                ),
            )
            uploaded_outputs.append(
                {
                    "asset_id": artifact.asset_id,
                    "name": artifact.name,
                    "role": artifact.role,
                    "format": artifact.file_format,
                    "sha256": artifact_metadata["output"]["sha256"],
                    "size_bytes": artifact_metadata["output"]["size_bytes"],
                }
            )

        error_payload = None
        if error is not None:
            error_payload = {
                "type": type(error).__qualname__,
                "message": str(error),
                "repr": _safe_repr(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        cell_payload = {
            **self._base_metadata("cell_execution"),
            "execution": {
                "cell_sequence": sequence,
                "started_at": started_at or completed_at,
                "completed_at": completed_at,
                "duration_seconds": round(duration_seconds, 9),
                "status": "succeeded" if success else "failed",
                "source_sha256": source_hash,
                "source": source,
                "stdout": stdout,
                "stderr": stderr,
                "error": error_payload,
                "rich_outputs": rich_output_manifest,
                "assigned_names": sorted(set(assigned_names)),
            },
            "output_assets": uploaded_outputs,
        }
        cell_record = cell_dir / "cell-execution.json"
        _write_json(cell_record, cell_payload)
        cell_asset_id = self._upload(
            cell_record,
            title=_title(
                "RHEED cell execution",
                self.notebook_relative_path,
                self.run_id,
                f"cell {sequence:04d}",
            ),
            description=(
                "Complete code-cell execution record, including source, streams, "
                "rich output manifest, status, and linked scientific artifacts."
            ),
            metadata={
                **self._base_metadata("cell_execution"),
                "execution": {
                    "cell_sequence": sequence,
                    "status": cell_payload["execution"]["status"],
                    "source_sha256": source_hash,
                    "started_at": cell_payload["execution"]["started_at"],
                    "completed_at": completed_at,
                    "duration_seconds": cell_payload["execution"]["duration_seconds"],
                    "output_asset_count": len(uploaded_outputs),
                },
            },
            tags=self._tags(
                "cell_execution",
                f"cell:{sequence:04d}",
                f"status:{cell_payload['execution']['status']}",
                "asset-type:notebook-cell",
            ),
        )
        self._link(
            cell_asset_id,
            self.run_asset_id,
            "part_of_run",
            qualifiers={"run_id": self.run_id, "cell_sequence": sequence},
        )
        if self.last_cell_asset_id:
            self._link(
                cell_asset_id,
                self.last_cell_asset_id,
                "continues_from",
                qualifiers={"cell_sequence": sequence},
            )
        else:
            self._link(
                cell_asset_id,
                self.notebook_asset_id,
                "starts_from_notebook",
                qualifiers={"cell_sequence": sequence},
            )

        for artifact in artifacts:
            self._link(
                artifact.asset_id,
                cell_asset_id,
                "generated_by",
                qualifiers={
                    "run_id": self.run_id,
                    "cell_sequence": sequence,
                    "output_name": artifact.name,
                    "output_role": artifact.role,
                    "format": artifact.file_format,
                },
            )
        semantic_order = {
            "configuration": 0,
            "raw_data": 1,
            "derived_data": 2,
            "analysis": 3,
        }
        for artifact in sorted(
            artifacts, key=lambda item: semantic_order.get(item.role, 99)
        ):
            self._link_semantic_provenance(artifact)

        for asset_id, relationship in self._pending_file_assets:
            self._link(asset_id, cell_asset_id, relationship, qualifiers={"cell_sequence": sequence})
        self._pending_file_assets.clear()
        self.last_cell_asset_id = cell_asset_id
        self.cell_count = sequence
        return cell_asset_id

    def _serialize_assigned_value(
        self, cell_dir: Path, name: str, value: Any
    ) -> PendingArtifact | None:
        import numpy as np
        if not isinstance(value, type) and callable(getattr(value, "get_weights", None)) and callable(getattr(value, "to_json", None)):
            path = cell_dir / f"{_slug(name)}.npz"
            np.savez_compressed(path, architecture_json=np.asarray(value.to_json()),
                                **{f"weight_{i}": w for i, w in enumerate(value.get_weights())})
            return PendingArtifact(path, name, "model", "npz", type(value).__qualname__,
                                   {"restore": "model_from_json with QKeras custom_objects, then set_weights in numeric order"})
        if hasattr(value, "history") and isinstance(value.history, dict):
            value = {"history": value.history, "params": getattr(value, "params", {}),
                     "epoch": getattr(value, "epoch", [])}
        if isinstance(value, Mapping) and value and all(isinstance(v, np.ndarray) for v in value.values()):
            path = cell_dir / f"{_slug(name)}.npz"
            arrays = {str(k): v for k, v in value.items() if not _SECRET_KEY.search(str(k))}
            np.savez_compressed(path, **arrays)
            return PendingArtifact(path, name, self._classify_output(name, value), "npz", "mapping",
                                   {"arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}})
        if type(value).__module__.startswith("tensorflow") and hasattr(value, "numpy"):
            value = value.numpy()
        if (
            isinstance(value, types.ModuleType)
            or isinstance(value, type)
            or callable(value)
        ):
            return None
        safe_name = _slug(name, limit=60)
        role = self._classify_output(name, value)
        python_type = f"{type(value).__module__}.{type(value).__qualname__}"

        try:
            import numpy as np
        except ImportError:  # pragma: no cover - RHEED notebooks require numpy
            np = None

        if np is not None and isinstance(value, np.ndarray):
            path = cell_dir / f"{safe_name}.npz"
            np.savez_compressed(path, value=value)
            return PendingArtifact(
                path=path,
                name=name,
                role=role,
                file_format="npz",
                python_type=python_type,
                details={"shape": list(value.shape), "dtype": str(value.dtype)},
            )

        if np is not None and isinstance(value, (list, tuple)):
            array_items = {
                f"item_{index}": item
                for index, item in enumerate(value)
                if isinstance(item, np.ndarray)
            }
            if array_items:
                path = cell_dir / f"{safe_name}.npz"
                np.savez_compressed(path, **array_items)
                return PendingArtifact(
                    path=path,
                    name=name,
                    role=role,
                    file_format="npz",
                    python_type=python_type,
                    details={
                        "arrays": {
                            key: {"shape": list(item.shape), "dtype": str(item.dtype)}
                            for key, item in array_items.items()
                        }
                    },
                )

        if type(value).__module__.startswith("pandas") and hasattr(value, "to_csv"):
            path = cell_dir / f"{safe_name}.csv"
            value.to_csv(path)
            return PendingArtifact(
                path=path,
                name=name,
                role=role,
                file_format="csv",
                python_type=python_type,
                details={
                    "shape": list(getattr(value, "shape", ())),
                    "columns": [str(item) for item in getattr(value, "columns", [])],
                },
            )

        if type(value).__module__.startswith("matplotlib") and hasattr(
            value, "savefig"
        ):
            path = cell_dir / f"{safe_name}.png"
            value.savefig(path, dpi=180, bbox_inches="tight")
            return PendingArtifact(
                path=path,
                name=name,
                role="analysis",
                file_format="png",
                python_type=python_type,
                details={"mime_type": "image/png"},
            )

        if isinstance(value, (bytes, bytearray, memoryview)):
            path = cell_dir / f"{safe_name}.bin"
            path.write_bytes(bytes(value))
            return PendingArtifact(
                path=path,
                name=name,
                role=role,
                file_format="binary",
                python_type=python_type,
            )

        if isinstance(value, Path) and value.is_file():
            path = cell_dir / f"{safe_name}-{value.name}"
            shutil.copy2(value, path)
            return PendingArtifact(
                path=path,
                name=name,
                role=role,
                file_format=value.suffix.lstrip(".") or "binary",
                python_type=python_type,
                details={"original_path": str(value)},
            )

        qick_payload: dict[str, Any] = {}
        if hasattr(value, "cfg"):
            qick_payload["cfg"] = _json_safe(getattr(value, "cfg"))
        dump_prog = getattr(value, "dump_prog", None)
        if callable(dump_prog):
            try:
                qick_payload["program"] = _json_safe(dump_prog())
            except Exception as exc:
                qick_payload["program_error"] = f"{type(exc).__name__}: {exc}"
        if qick_payload:
            qick_payload["python_type"] = python_type
            path = cell_dir / f"{safe_name}.json"
            _write_json(path, qick_payload)
            return PendingArtifact(
                path=path,
                name=name,
                role="configuration",
                file_format="json",
                python_type=python_type,
            )

        if isinstance(value, (Mapping, list, tuple, set, frozenset)) and role in {
            "configuration",
            "raw_data",
            "analysis",
            "derived_data",
        }:
            path = cell_dir / f"{safe_name}.json"
            _write_json(path, value)
            return PendingArtifact(
                path=path,
                name=name,
                role=role,
                file_format="json",
                python_type=python_type,
                details=(
                    {"keys": [str(item) for item in list(value)[:500]]}
                    if isinstance(value, Mapping)
                    else {"length": len(value)}
                ),
            )
        return None

    @staticmethod
    def _classify_output(name: str, value: Any) -> str:
        lower = name.lower()
        if type(value).__module__.startswith("matplotlib") or re.search(
            r"(?:^|_)(?:fig|figure|plot|hist|fit|model|result|popt|curve)(?:$|_)",
            lower,
        ):
            return "analysis"
        if re.search(r"(?:^|_)(?:cfg|config|soccfg|prog|program|pulse)(?:$|_)", lower):
            return "configuration"
        if re.search(
            r"(?:iq|avgi|avgq|shot|trace|signal|data|freqs?|times?|amps?|phases?|expt_pts)",
            lower,
        ):
            return "raw_data"
        return "derived_data"

    def _link_semantic_provenance(self, artifact: PendingArtifact) -> None:
        previous_same_role = self._latest_role_asset.get(artifact.role)
        if artifact.role == "configuration" and previous_same_role:
            self._link(
                artifact.asset_id,
                previous_same_role,
                "revises_configuration",
                qualifiers={"output_name": artifact.name},
            )
        elif artifact.role == "raw_data":
            configuration = self._latest_role_asset.get("configuration")
            if configuration:
                self._link(
                    artifact.asset_id,
                    configuration,
                    "acquired_with",
                    qualifiers={"output_name": artifact.name},
                )
        elif artifact.role == "analysis":
            raw_data = self._latest_role_asset.get("raw_data")
            if raw_data:
                self._link(
                    artifact.asset_id,
                    raw_data,
                    "analysis_of",
                    qualifiers={"output_name": artifact.name},
                    qualifier_note=(
                        "Notebook analysis is non-destructive to the stored "
                        "upstream acquisition."
                    ),
                    analysis_mode="non_destructive",
                )
        self._latest_role_asset[artifact.role] = artifact.asset_id

    def capture_file(self, path: str | Path, *, role: str = "analysis",
                     relationship: str = "generated_by", metadata: dict | None = None) -> str:
        """Persist an explicitly selected input/output; no directory-wide secret scanning."""
        original = Path(path).resolve()
        digest = _sha256_bytes(original.read_bytes())
        target = self.run_dir / "files" / digest / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original != target:
            shutil.copy2(original, target)
        details = self._base_metadata("executed_notebook" if role == "executed_notebook" else "file_artifact")
        details["output"] = {"name": original.name, "role": role,
                             "format": original.suffix.lstrip("."), "sha256": digest,
                             "size_bytes": target.stat().st_size, **(metadata or {})}
        asset_id = self._upload(target, title=_title("RHEED file", self.repo_root.name, self.run_id, original.name, digest[:12]),
                                description=f"{role} for RHEED run {self.run_id}", metadata=details,
                                tags=self._tags("file_artifact", role))
        if relationship == "uses_dependency":
            self._link(self.run_asset_id, asset_id, relationship, qualifiers={"sha256": digest})
        elif self._executing_cell:
            self._link(asset_id, self.run_asset_id, "part_of_run", qualifiers={"run_id": self.run_id})
            self._pending_file_assets.append((asset_id, relationship))
        else:
            self._link(asset_id, self.run_asset_id, relationship,
                       qualifiers={"run_id": self.run_id, "sha256": digest})
        return asset_id

    def finish(self, *, status: str | None = None) -> str | None:
        """Upload a terminal summary record and close the Dataerai client."""

        if not self.running:
            return self.summary_asset_id
        self.finished_at = _utc_now()
        status = "failed" if self.execution_errors or status == "failed" else ("completed_with_capture_errors" if self.errors else status or "completed")
        summary = {
            **self._base_metadata("notebook_run_summary"),
            "run_summary": {
                "status": status,
                "finished_at": self.finished_at,
                "cell_count": self.cell_count,
                "asset_count_before_summary": self.asset_count,
                "relationship_count_before_summary": self.relationship_count,
                "capture_errors": self.errors,
                "execution_errors": self.execution_errors,
            },
        }
        summary_path = self.run_dir / "run-summary.json"
        _write_json(summary_path, summary)
        try:
            self.summary_asset_id = self._upload(
                summary_path,
                title=_title(
                    "RHEED notebook run summary",
                    self.notebook_relative_path,
                    self.run_id,
                ),
                description=(
                    "Terminal summary for a continuous RHEED notebook provenance run."
                ),
                metadata=summary,
                tags=self._tags(
                    "notebook_run_summary",
                    f"status:{status}",
                    "asset-type:run-summary",
                ),
            )
            self._link(
                self.summary_asset_id,
                self.run_asset_id,
                "summarizes",
                qualifiers={"run_id": self.run_id, "cell_count": self.cell_count},
            )
            if self.last_cell_asset_id:
                self._link(
                    self.summary_asset_id,
                    self.last_cell_asset_id,
                    "continues_from",
                    qualifiers={"terminal_record": True},
                )
            updated_run_metadata = {
                **self._run_metadata,
                "completion": {
                    "status": status,
                    "finished_at": self.finished_at,
                    "cell_count": self.cell_count,
                    "asset_count": self.asset_count,
                    "relationship_count": self.relationship_count,
                    "summary_asset_id": self.summary_asset_id,
                },
            }
            self.client.set_metadata(self.run_asset_id, metadata=updated_run_metadata)
        finally:
            self.running = False
            if self._owns_client:
                self.client.close()
        print(
            f"Dataerai provenance finalized: {self.cell_count} cells, "
            f"{self.asset_count} assets, {self.relationship_count} relationships\n"
            f"  summary asset: {self.summary_asset_id}"
        )
        return self.summary_asset_id

    close = finish

    def __repr__(self) -> str:
        state = "active" if self.running else "inactive"
        return (
            f"NotebookProvenance(state={state!r}, run_id={self.run_id!r}, "
            f"notebook={self.notebook_relative_path!r}, cells={self.cell_count})"
        )


def start_dataerai_notebook(
    notebook_path: str | Path,
    **kwargs: Any,
) -> NotebookProvenance:
    """Create, authenticate, and start a provenance-tracked notebook run."""

    return NotebookProvenance(notebook_path, **kwargs).start()


__all__ = [
    "DisplayOutput",
    "NotebookProvenance",
    "SCHEMA",
    "start_dataerai_notebook",
]
