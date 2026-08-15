# SDK releases

The `vX.Y.Z` tag is the release boundary for the `haliosai` distribution. A tag runs
package validation, installs the built wheel on Python 3.10–3.14 across Linux, macOS, and Windows,
smoke-tests the explicit public imports, and publishes the same artifacts to PyPI.

## One-time PyPI setup

Configure a PyPI trusted publisher for this repository, workflow `sdk-release.yml`, environment
`pypi`, and package `haliosai`. Protect the GitHub `pypi` environment so only approved release
maintainers can publish.

## Release

1. Update `haliosai/_version.py` and the changelog/release notes.
2. Run the SDK test and package checks locally.
3. Create and push a tag whose version exactly matches `_version.py`:

   ```bash
   git tag -s v2.0.0 -m "Halios SDK 2.0.0"
   git push origin v2.0.0
   ```

The workflow rejects mismatched tags and versions before publishing. PyPI versions are immutable;
fixes require a new patch version and tag.
