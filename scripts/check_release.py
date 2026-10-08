#!/usr/bin/env python3
"""Audit public source/artifacts without printing matched private values.

This is a publication guard, not a universal secret detector. It deliberately publishes
only source, synthetic fixtures and docs; operators must also review release contents.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".gitignore",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
}
FIXTURES = {"tests/fixtures/torch.json", "tests/fixtures/vllm.prom", "tests/fixtures/sglang.prom"}
VENDOR_DATA = {"src/obsrv/_vendor/provenance.json", "src/obsrv/_vendor/VIBESYS_LICENSE"}
RUNTIME_PARTS = {
    ".git",
    ".venv",
    "venv",
    ".obsrv",
    "state",
    "artifacts",
    "exports",
    "traces",
    "logs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}
RULES = {
    "private home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+(?:/|\b)"),
    "email address": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "API token": re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,})\b"),
    "AWS key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential URL": re.compile(r"https?://[^\s/'\"]+:[^\s/@'\"]+@"),
}


def allowed_source(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(p in RUNTIME_PARTS for p in path.parts):
        return False
    if name in ROOT_FILES | FIXTURES | VENDOR_DATA:
        return True
    if name.startswith("src/obsrv/") and path.suffix == ".py":
        return True
    if name.startswith("tests/") and path.suffix == ".py":
        return True
    if name.startswith("scripts/") and path.suffix == ".py":
        return True
    if name.startswith("docs/") and path.suffix == ".md":
        return True
    return name == "examples/deployment.toml"


def findings(name, data):
    text = data.decode("utf-8", errors="replace")
    # No value, snippet, absolute local path or token is included in diagnostics.
    return [f"{name}: {label}" for label, pattern in RULES.items() if pattern.search(text)]


def source_files(root):
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if allowed_source(relative):
            if path.is_symlink() or root.resolve() not in path.resolve().parents:
                raise ValueError("publication source cannot contain symlinks")
            if path.is_file():
                files.append(path)
    return sorted(files)


def check_source(root):
    errors = []
    files = source_files(root)
    for path in files:
        errors.extend(findings(path.relative_to(root).as_posix(), path.read_bytes()))
    # Runtime files must not already be tracked/staged, even if Git now ignores them.
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False
    )
    if tracked.returncode == 0:
        for name in tracked.stdout.decode().split("\0"):
            if name and not allowed_source(name):
                errors.append("tracked file is outside the publication allowlist: " + name)
    return files, errors


def _archive_items(path):
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                if item.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError("archive cannot contain symlinks")
                yield item.filename, archive.read(item)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for item in archive.getmembers():
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("source archive cannot contain non-regular files")
                with archive.extractfile(item) as file:
                    yield item.name, file.read()
    else:
        raise ValueError("unsupported release archive")


def check_archive(path):
    errors = []
    count = 0
    names = set()
    for name, data in _archive_items(path):
        count += 1
        if name in names:
            errors.append(f"{path.name}: duplicate member")
        names.add(name)
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
            errors.append(f"{path.name}: unsafe member path")
            continue
        if path.suffix == ".whl":
            if name.startswith("obsrv/"):
                allowed = allowed_source("src/" + name)
            elif parts[0].endswith(".dist-info"):
                allowed = "/".join(parts[1:]) in {
                    "METADATA",
                    "WHEEL",
                    "RECORD",
                    "entry_points.txt",
                    "licenses/LICENSE",
                    "licenses/NOTICE",
                }
            else:
                allowed = False
        elif path.name.endswith(".tar.gz"):
            relative = "/".join(parts[1:])
            allowed = allowed_source(relative) or relative == "PKG-INFO"
        else:
            allowed = allowed_source(name)
        if not allowed:
            errors.append(f"{path.name}: unexpected member {name}")
        errors.extend(findings(f"{path.name}:{name}", data))
    if count == 0:
        errors.append(f"{path.name}: empty archive")
    return count, errors


def source_bundle(root, files, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
    return check_archive(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archives", type=Path, nargs="*", default=[])
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--inventory", type=Path, help="write reviewed source names and digests")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    files, errors = check_source(root)
    counts = {}
    for archive in args.archives:
        count, issues = check_archive(archive)
        counts[archive.name] = count
        errors.extend(issues)
    if errors:
        print("Publication audit failed (matched values are withheld):")
        for error in errors:
            print("- " + error)
        return 1
    if args.source_bundle:
        count, issues = source_bundle(root, files, args.source_bundle)
        counts[args.source_bundle.name] = count
        if issues:
            args.source_bundle.unlink(missing_ok=True)
            print("Source bundle audit failed; bundle removed.")
            return 1
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(
            json.dumps(
                {
                    "source": [
                        {
                            "path": p.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        }
                        for p in files
                    ],
                    "archives": counts,
                    "scope": "source allowlist and built archives; no private values printed",
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Publication audit passed: {len(files)} source files; {len(counts)} archives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
