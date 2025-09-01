.PHONY: help install install-dev test lint format clean build publish-test publish

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e .[dev]

install-all:  ## Install package with all optional dependencies
	pip install -e .[all,dev]

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=haliosai --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 haliosai/ tests/ examples/
	mypy haliosai/

format:  ## Format code
	black haliosai/ tests/ examples/
	isort haliosai/ tests/ examples/

format-check:  ## Check code formatting
	black --check haliosai/ tests/ examples/
	isort --check-only haliosai/ tests/ examples/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

check-build:  ## Check package build
	python -m twine check dist/*

publish-test:  ## Publish to test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

example:  ## Run basic example
	cd examples && python basic_usage.py

setup-dev:  ## Set up development environment
	python -m venv venv
	@echo "Please run: source venv/bin/activate (or venv\\Scripts\\activate on Windows)"
	@echo "Then run: make install-dev"

deps-check:  ## Check for dependency updates
	pip list --outdated

security-check:  ## Run security checks
	pip-audit

all-checks: format-check lint test  ## Run all checks

# Release workflow
prepare-release: clean all-checks build check-build  ## Prepare for release

# Development workflow  
dev-setup: setup-dev install-dev  ## Complete development setup
