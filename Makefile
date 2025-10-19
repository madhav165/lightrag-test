# Makefile for common dev tasks

.PHONY: help install-hooks precommit clean

help:
	@echo "make install-hooks   - install pre-commit hooks"
	@echo "make precommit      - run pre-commit over all files"
	@echo "make clean          - remove python build artifacts"

install-hooks:
	uv pip install --upgrade pip
	uv pip install pre-commit
	pre-commit install

precommit:
	pre-commit run --all-files

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__
