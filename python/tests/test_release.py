from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("leo_release", ROOT / "scripts" / "release.py")
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseAutomationTests(unittest.TestCase):
    def test_version_normalization_accepts_optional_v_and_rejects_nonstable_versions(self) -> None:
        self.assertEqual(release.normalize_version("1.2.3"), "1.2.3")
        self.assertEqual(release.normalize_version("v1.2.3"), "1.2.3")
        for invalid in ("1.2", "1.2.3-beta.1", "01.2.3", "latest"):
            with self.assertRaises(release.ReleaseError):
                release.normalize_version(invalid)

    def test_changelog_rollover_preserves_unreleased_and_release_notes(self) -> None:
        source = "# Changelog\n\n## Unreleased\n\n- Faster training.\n- Better docs.\n\n## v1.0.1 - 2026-09-07\n\n- Previous.\n"
        updated = release.release_changelog(source, "1.0.2", "2026-09-11")
        self.assertIn("## Unreleased\n\n## v1.0.2 - 2026-09-11", updated)
        self.assertIn("- Faster training.\n- Better docs.", updated)
        self.assertIn("## v1.0.1 - 2026-09-07", updated)
        self.assertEqual(
            release.changelog_notes(updated, "1.0.2"),
            "- Faster training.\n- Better docs.\n",
        )

    def test_prepare_updates_only_software_release_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "Cargo.toml").write_text(
                '[workspace]\n\n[workspace.package]\nversion = "1.0.1"\n',
                encoding="utf-8",
            )
            (root / "CITATION.cff").write_text(
                'title: "Leo"\nversion: "1.0.1"\n', encoding="utf-8"
            )
            (root / "docs" / "SEMANTICS.md").write_text(
                "| Leo release | 1.0.1 |\n| Model schema | 1 |\n",
                encoding="utf-8",
            )
            (root / "CHANGELOG.md").write_text(
                "# Changelog\n\n## Unreleased\n\n- New work.\n\n## v1.0.1 - 2026-09-07\n\n- Old work.\n",
                encoding="utf-8",
            )

            release.prepare(root, "1.0.2", "2026-09-11")

            self.assertEqual(release.current_version(root), "1.0.2")
            self.assertIn(
                'version: "1.0.2"',
                (root / "CITATION.cff").read_text(encoding="utf-8"),
            )
            self.assertIn(
                'date-released: "2026-09-11"',
                (root / "CITATION.cff").read_text(encoding="utf-8"),
            )
            semantics = (root / "docs" / "SEMANTICS.md").read_text(encoding="utf-8")
            self.assertIn("| Leo release | 1.0.2 |", semantics)
            self.assertIn("| Model schema | 1 |", semantics)
            self.assertIn(
                "## v1.0.2 - 2026-09-11",
                (root / "CHANGELOG.md").read_text(encoding="utf-8"),
            )

    def test_workflow_uses_protected_pr_check_merge_tag_release_flow(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        for required in (
            "workflow_dispatch:",
            "RELEASE_TOKEN",
            "gh pr create",
            "gh pr checks",
            "gh pr merge",
            "--squash",
            "git tag -a",
            "gh release create",
        ):
            self.assertIn(required, workflow)


if __name__ == "__main__":
    unittest.main()
