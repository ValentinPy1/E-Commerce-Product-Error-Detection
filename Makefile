.PHONY: help install install-dev test test-cov lint format type-check clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .

test: ## Run tests
	python3 -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python3 -m pytest tests/ --cov=models --cov=scripts --cov-report=term-missing --cov-report=html

lint: ## Run linters
	python3 -m ruff check .
	python3 -m black --check .

format: ## Format code
	python3 -m black .
	python3 -m ruff check --fix .

type-check: ## Run type checker
	python3 -m mypy models/ scripts/ --ignore-missing-imports

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf htmlcov/
	rm -rf .coverage

pre-commit: ## Install pre-commit hooks
	python3 -m pre_commit install

pre-commit-run: ## Run pre-commit on all files
	python3 -m pre_commit run --all-files

