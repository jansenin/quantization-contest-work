"""Tests for the restartable model-download orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import download_models


class DownloadModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = download_models.load_catalog(download_models.DEFAULT_CATALOG)

    def test_default_selection_is_laptop_profile(self) -> None:
        selected = download_models.select_models(self.catalog, [], [])
        self.assertEqual(selected, self.catalog["profiles"]["laptop"])

    def test_profiles_and_explicit_models_are_deduplicated(self) -> None:
        selected = download_models.select_models(
            self.catalog, ["laptop", "genuine"], ["qwen3-0.6b"]
        )
        self.assertEqual(selected.count("qwen3-0.6b"), 1)
        self.assertIn("qwen3-0.6b-nvfp4", selected)

    def test_unknown_selection_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model aliases"):
            download_models.select_models(self.catalog, [], ["not-a-model"])

    def test_state_write_is_valid_and_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "state.json"
            state = {"version": 1, "models": {"example": {"status": "resolved"}}}
            download_models.save_state(path, state)
            self.assertEqual(download_models.load_state(path), state)
            state["models"]["example"]["status"] = "complete"
            download_models.save_state(path, state)
            self.assertEqual(json.loads(path.read_text()), state)

    def test_resolution_outage_retries_and_pins_revision(self) -> None:
        class FlakyApi:
            calls = 0

            def model_info(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("offline")
                return SimpleNamespace(sha="a" * 40)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            state_path = root / "state.json"
            state = {"version": 1, "models": {}}
            api = FlakyApi()
            with mock.patch.object(download_models.time, "sleep"):
                download_models._download_one(
                    alias="example",
                    spec={"repo_id": "org/model", "revision": "main"},
                    allow_patterns=["*.json"],
                    cache_dir=root / "cache",
                    state_path=state_path,
                    state=state,
                    api=api,
                    snapshot_download=lambda **_kwargs: str(snapshot),
                    permanent_errors=(ValueError,),
                    workers=1,
                    initial_delay=1,
                    max_delay=2,
                    max_retries=2,
                )
            record = state["models"]["example"]
            self.assertEqual(api.calls, 2)
            self.assertEqual(record["resolved_revision"], "a" * 40)
            self.assertEqual(record["status"], "complete")


if __name__ == "__main__":
    unittest.main()
