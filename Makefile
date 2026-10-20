.PHONY: install test lint format api

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

api:
	uvicorn src.api.main:app --reload
