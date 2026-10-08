# Release checklist

Publication is an explicit operator action. This repository's CI builds and checks
packages but does not create GitHub repositories, push code, release tags or upload to PyPI.

## Prepare

1. Review the README, license/NOTICE and analyzer provenance. Keep the alpha label until
   the documented real-deployment validation is complete. Update CHANGELOG for changes.
2. Work from a clean checkout and a fresh environment. Install development dependencies.
3. Run tests, lint and formatting; build the wheel/source archive.
4. Run the release checker against source and archives. Review its file inventory and
   inspect Git's tracked/staged files and history for confidential content. Do not publish
   runtime directories, real traces, private configs or another project's Git history.
5. Install the wheel into a separate clean environment. Check CLI, analyzer resources and
   offline trace import/export/verification. The install should not need a sibling repository.

```bash
python -m pytest
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/check_release.py

python -m pip install build
python -m build
python scripts/check_release.py --archives dist/*.whl dist/*.tar.gz
```

Alternatively, `uv build` uses the configured build backend.
The source-release checker can produce a portable source bundle without `.git`, local
metadata or runtime state:

```bash
python scripts/check_release.py --source-bundle dist/obsrv-source.zip
```

Record checksums and release contents. Synthetic privacy canaries are exercised in tests;
no real secret is needed for validation. The scanner never prints matched values.

## Publish deliberately

Choose the GitHub owner/repository and visibility, then push the reviewed source only.
Use a fresh repository/history for this package. Review commit author/email metadata,
particularly when publishing from a machine with a private default Git identity.

Once CI passes, tag a release and attach only the audited wheel, source distribution
and optional source bundle. Check release asset digests against the local audited files.
Update installation links with the actual repository URL after it exists; this project
currently gives clone-root/release-wheel instructions without inventing an owner or URL.

PyPI publication is a separate action. Verify project-name availability and ownership
before using `pip install obsrv` in documentation; the package is not published there.
