import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_release.py"
spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_runtime_canaries_excluded_from_source_bundle(tmp_path):
    (tmp_path / "README.md").write_text("Public source")
    for name in (
        ".obsrv/deployment/obsrv.json",
        ".env",
        "state/raw.sqlite",
        "exports/context.md",
        ".venv/pyvenv.cfg",
        "artifacts/trace.json.gz",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime-only-canary")
    files, errors = audit.check_source(tmp_path)
    assert errors == []
    output = tmp_path / "release.zip"
    _, errors = audit.source_bundle(tmp_path, files, output)
    assert errors == []
    with zipfile.ZipFile(output) as bundle:
        assert bundle.namelist() == ["README.md"]
        assert b"runtime-only-canary" not in bundle.read("README.md")


@pytest.mark.parametrize(
    "value, label",
    [
        ("/" + "Users/" + "someone/" + "private", "private home path"),
        ("gh" + "p_" + "a" * 36, "GitHub token"),
        ("sk-" + "a" * 48, "API token"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private key"),
        ("http://" + "user:password@" + "example.test", "credential URL"),
        ("someone" + "@" + "example.test", "email address"),
    ],
)
def test_private_patterns_report_category_without_value(value, label):
    findings = audit.findings("example.txt", value.encode())
    assert "example.txt: " + label in findings
    assert value not in "\n".join(findings)


def test_archive_audit_rejects_runtime_and_private_content(tmp_path):
    path = tmp_path / "bad.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("obsrv/state/obsrv.json", "runtime-only-canary")
        archive.writestr("obsrv/main.py", "/" + "Users/" + "someone/" + "private")
    _, errors = audit.check_archive(path)
    assert any("unexpected member" in error for error in errors)
    assert any("private home path" in error for error in errors)
    assert "runtime-only-canary" not in "\n".join(errors)


def test_archive_audit_rejects_symlink(tmp_path):
    path = tmp_path / "link.zip"
    entry = zipfile.ZipInfo("README.md")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(entry, "outside-source")
    with pytest.raises(ValueError, match="symlinks"):
        audit.check_archive(path)


def test_packaged_analyzers_match_manifest():
    import hashlib

    root = Path(__file__).parents[1] / "src/obsrv/_vendor"
    manifest = json.loads((root / "provenance.json").read_text())
    for source, name in [
        ("torch/analyze_torch_profile.py", "torch.py"),
        ("nsys/analyze_nsys.py", "nsys.py"),
    ]:
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            == (manifest["files"][source]["vendored_sha256"])
        )
