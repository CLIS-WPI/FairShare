.PHONY: help install test lint format clean docker-build docker-run plots notebook

help:
	@echo "Fuzzy-Fairness DSS LEO - Makefile Commands"
	@echo ""
	@echo "  make install      - Install package and dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run linters (black, flake8, isort)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make plots        - Generate all plots"
	@echo "  make notebook     - Start Jupyter Lab"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	black --check src/ tests/ experiments/
	flake8 src/ tests/ experiments/ --max-line-length=120
	isort --check-only src/ tests/ experiments/

format:
	black src/ tests/ experiments/
	isort src/ tests/ experiments/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:latest .

docker-run:
	docker run --gpus all -it -v $(PWD):/workspace --name fuzzy-dss-dev fuzzy-fairness-dss:latest /bin/bash

docker-run-cpu:
	docker run -it -v $(PWD):/workspace --name fuzzy-dss-dev fuzzy-fairness-dss:latest /bin/bash

docker-stop:
	docker stop fuzzy-dss-dev || true
	docker rm fuzzy-dss-dev || true

plots:
	python experiments/generate_plots.py --scenario urban_congestion_phase4
	python experiments/generate_plots.py --scenario rural_coverage_phase4
	python experiments/generate_plots.py --scenario emergency_response_phase4

notebook:
	jupyter lab notebooks/interactive_demo.ipynb

