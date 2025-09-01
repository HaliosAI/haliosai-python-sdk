# Contributing to HaliosAI SDK

We love your input! We want to make contributing to HaliosAI SDK as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/halioslabs/haliosai-sdk.git
cd haliosai-sdk
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e .[dev]
```

4. Run tests:
```bash
pytest
```

5. Run linting:
```bash
black haliosai/
isort haliosai/
flake8 haliosai/
mypy haliosai/
```

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/halioslabs/haliosai-sdk/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/halioslabs/haliosai-sdk/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports.

## Use a Consistent Coding Style

* We use [Black](https://github.com/psf/black) for code formatting
* We use [isort](https://github.com/PyCQA/isort) for import sorting
* We follow PEP 8 with line length of 100 characters
* We use type hints throughout the codebase

## Testing

* Write tests for any new functionality
* Ensure all tests pass before submitting a PR
* Use pytest for testing
* Include both unit tests and integration tests where appropriate

## Documentation

* Update docstrings for any changed functions/classes
* Update README.md if you change functionality
* Update CHANGELOG.md following the existing format
* Use clear, concise language in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
