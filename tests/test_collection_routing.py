from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataerai_console_api import DataeraiConsoleAPI
from dataerai_notebook import NotebookProvenance
from tests.helpers import COLLECTION_ID, PROJECT_ID, FakeClient, FakeConsoleAPI, FakeShell


def test_collection_title_distinguishes_repository_notebook_and_affixes(tmp_path):
    repo = tmp_path / "Rheed-Folo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "Generated_Flow.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}')
    console = FakeConsoleAPI()

    tracker = NotebookProvenance(
        notebook,
        client=FakeClient(),
        shell=FakeShell(),
        console_api=console,
        collection_prefix="September batch",
        collection_postfix="rerun 2",
    ).start()

    assert console.collection_requests == [{
        "owner_type": "user",
        "owner_id": "11111111-1111-4111-8111-111111111111",
        "title": "September batch · Rheed-Folo · Generated_Flow · rerun 2",
        "description": "Dataerai provenance records for Rheed-Folo/Generated_Flow.ipynb.",
        "tags": ["rheed", "notebook-runs", "repository:rheed-folo", "notebook:generated-flow"],
    }]
    assert tracker.collection_id == COLLECTION_ID
    assert tracker.owner_type == "project"
    assert tracker.owner_id == PROJECT_ID
    assert tracker.collection_resolution == "notebook_collection_created_or_reused"


def test_explicit_collection_id_bypasses_automatic_collection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}')
    console = FakeConsoleAPI()

    tracker = NotebookProvenance(
        notebook,
        client=FakeClient(),
        shell=FakeShell(),
        console_api=console,
        collection_id=COLLECTION_ID,
        collection_prefix="ignored",
    ).start()

    assert console.collection_requests == []
    assert tracker.collection_resolution == "explicit"


def test_get_or_create_reuses_exact_collection_and_creates_distinct_postfix():
    api = object.__new__(DataeraiConsoleAPI)
    requests = []
    existing = {
        "id": COLLECTION_ID,
        "title": "Rheed-Folo · Generated_Flow",
        "owner_project_id": PROJECT_ID,
        "owner_user_id": None,
    }

    def request(method, path, **kwargs):
        requests.append((method, path, kwargs))
        if method == "GET":
            title = kwargs["query"]["q"]
            return [existing] if title == existing["title"] else []
        return {
            "id": "44444444-4444-4444-8444-444444444444",
            "title": kwargs["payload"]["title"],
            "owner_project_id": PROJECT_ID,
            "owner_user_id": None,
        }

    api._request = request
    reused = api.get_or_create_notebook_collection(
        owner_type="project", owner_id=PROJECT_ID,
        title=existing["title"], description="records", tags=["rheed"],
    )
    created = api.get_or_create_notebook_collection(
        owner_type="project", owner_id=PROJECT_ID,
        title=existing["title"] + " · rerun", description="records", tags=["rheed"],
    )

    assert reused["id"] == COLLECTION_ID
    assert reused["owner_type"] == "project"
    assert reused["owner_id"] == PROJECT_ID
    assert created["title"].endswith("· rerun")
    assert created["owner_type"] == "project"
    assert created["owner_id"] == PROJECT_ID
    assert [method for method, _, _ in requests] == ["GET", "GET", "POST"]
    create_payload = requests[-1][2]["payload"]
    assert create_payload["owner_type"] == "project"
    assert create_payload["owner_id"] == PROJECT_ID
    assert "parent_id" not in create_payload


@pytest.mark.parametrize("value", ["/", "\x00", " " * 40])
def test_invalid_collection_affix_is_rejected(tmp_path, value):
    notebook = tmp_path / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}')
    with pytest.raises(ValueError, match="collection prefix"):
        NotebookProvenance(
            notebook, client=FakeClient(), shell=FakeShell(),
            console_api=FakeConsoleAPI(), collection_prefix=value,
        )


def test_conflicting_postfix_environment_aliases_are_rejected(tmp_path, monkeypatch):
    notebook = tmp_path / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}')
    monkeypatch.setenv("DATAERAI_COLLECTION_POSTFIX", "one")
    monkeypatch.setenv("DATAERAI_COLLECTION_SUFFIX", "two")
    with pytest.raises(ValueError, match="conflict"):
        NotebookProvenance(
            notebook,
            client=FakeClient(),
            shell=FakeShell(),
            console_api=FakeConsoleAPI(),
        )
