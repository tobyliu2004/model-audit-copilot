.PHONY: help install install-dev test test-cov lint format clean build docker-build docker-run docs

help:
	@echo "Model Audit Copilot - Development Commands"
	@echo ""
	@echo "install        Install package in production mode"
	@echo "install-dev    Install package in development mode with all dependencies"
	@echo "test           Run tests"
	@echo "test-cov       Run tests with coverage report"
	@echo "lint           Run linting checks"
	@echo "format         Format code with black and isort"
	@echo "clean          Remove build artifacts and cache files"
	@echo "build          Build distribution packages"
	@echo "docker-build   Build Docker image"
	@echo "docker-run     Run Docker container"
	@echo "docs           Build documentation"

install:
	pip install -r requirements.txt
	pip install .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=copilot --cov-report=html --cov-report=term

lint:
	flake8 copilot/ scripts/ tests/
	mypy copilot/
	black --check copilot/ scripts/ tests/
	isort --check-only copilot/ scripts/ tests/

format:
	black copilot/ scripts/ tests/
	isort copilot/ scripts/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docker-build:
	docker build -t model-audit-copilot .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data model-audit-copilot

docs:
	cd docs && make html

# Development shortcuts
run-dashboard:
	streamlit run dashboard/main_app.py

run-example:
	cd examples && python quickstart.py

generate-data:
	cd examples && python generate_sample_data.py

# Testing shortcuts
test-drift:
	pytest tests/test_drift_detector.py -v

test-fairness:
	pytest tests/test_fairness_audit.py -v

test-leakage:
	pytest tests/test_leakage_detector.py -v

# Git hooks
pre-commit:
	pre-commit run --all-files