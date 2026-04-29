.PHONY: install lint train test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check src tests

train:
	python src/train.py

test:
	pytest -q

clean:
	rm -rf artifacts/model artifacts/metrics.json mlruns .pytest_cache
