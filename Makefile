.PHONY: build publish publish-test test lint format clean

build:
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

test:
	PYTHONPATH=. pytest tests/ -v

lint:
	ruff check haliosai/ tests/
	mypy haliosai/

format:
	ruff format haliosai/ tests/

clean:
	rm -rf dist/ build/ *.egg-info haliosai/*.egg-info
