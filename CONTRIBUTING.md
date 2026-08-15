# Contributing

Thank you for improving the Halios Python SDK.

1. Open an issue for substantial public-API changes.
2. Create a focused branch with tests for behavior changes.
3. Run `python -m pip install -e '.[dev]'` and `python -m pytest -q`.
4. Run `python -m build` and `python -m twine check dist/*` before release changes.
5. Open a pull request describing compatibility and verification.

Never commit credentials, customer data, or private Halios service implementation details. The SDK
must remain a small explicit client and must not configure application tracing implicitly.
