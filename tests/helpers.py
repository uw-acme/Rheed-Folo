from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataerai_notebook import DisplayOutput, NotebookProvenance, SCHEMA


USER_ID = "11111111-1111-4111-8111-111111111111"
PROJECT_ID = "22222222-2222-4222-8222-222222222222"
COLLECTION_ID = "33333333-3333-4333-8333-333333333333"


class FakeShell:
    def __init__(self) -> None:
        self.user_ns = {}
        self.registered = None
        self.kernel = None

    def register_magic_function(self, function, *, magic_kind, magic_name):
        self.registered = (function, magic_kind, magic_name)

    @staticmethod
    def transform_cell(source: str) -> str:
        return source


class FakeClient:
    def __init__(self, *, user_id: str | None = USER_ID) -> None:
        self.uploads = []
        self.relationships = []
        self.metadata_updates = []
        self.record_types = {}
        self.projects = []
        self.closed = False
        self.user_id = user_id

    def connect(self):
        return None

    def auth_status(self):
        values = {"user_email": "student@example.edu"}
        if self.user_id is not None:
            values["user_id"] = self.user_id
        return SimpleNamespace(**values)

    def create_project(self, name, **kwargs):
        self.projects.append((name, kwargs))
        return SimpleNamespace(project_id=PROJECT_ID)

    def upload(self, path, **kwargs):
        asset_id = f"asset-{len(self.uploads) + 1}"
        self.uploads.append(
            {
                "asset_id": asset_id,
                "path": Path(path),
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(asset_id=asset_id)

    def create_relationship(self, from_asset_id, to_asset_id, **kwargs):
        self.relationships.append((from_asset_id, to_asset_id, kwargs))
        return SimpleNamespace(id=f"rel-{len(self.relationships)}")

    def set_metadata(self, asset_id, **kwargs):
        self.metadata_updates.append((asset_id, kwargs))

    def set_record_type(self, asset_id, *, record_type):
        self.record_types[asset_id] = record_type

    def close(self):
        self.closed = True


class FakeConsoleAPI:
    def __init__(self) -> None:
        self.collection_requests = []

    def get_or_create_notebook_collection(self, **kwargs):
        self.collection_requests.append(kwargs)
        return {
            "id": COLLECTION_ID,
            "title": kwargs["title"],
            "owner_type": "project",
            "owner_id": PROJECT_ID,
        }


def make_tracker(tmp_path: Path, **tracker_kwargs):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    client = FakeClient()
    shell = FakeShell()
    console_api = tracker_kwargs.pop("console_api", FakeConsoleAPI())
    tracker = NotebookProvenance(
        notebook,
        client=client,
        shell=shell,
        console_api=console_api,
        spool_root=repo / ".dataerai" / "runs",
        **tracker_kwargs,
    ).start()
    return tracker, client, shell

