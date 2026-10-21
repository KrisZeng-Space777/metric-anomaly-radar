.PHONY: help install demo test lint fmt clean

help:
	@echo "install : editable install with dev extras"
	@echo "demo    : synthetic data -> detection -> attribution -> figures"
	@echo "test    : pytest"
	@echo "lint    : ruff"
	@echo "clean   : remove generated data and results"

install:
	python -m pip install -e ".[dev]"

demo:
	mar demo --out .

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

clean:
	rm -rf results data/*.csv data/*.json _smoke .pytest_cache .ruff_cache
