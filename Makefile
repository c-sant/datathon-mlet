.PHONY: install format lint test run check precommit-install docker-build docker-up docker-down

install:
	pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .

test:
	pytest

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

check: lint test

precommit-install:
	pre-commit install
	pre-commit install --hook-type commit-msg

docker-build:
	docker build -t mlet-final-project:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down