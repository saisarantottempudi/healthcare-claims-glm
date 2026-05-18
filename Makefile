.PHONY: install data train evaluate serve test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/generate_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/**/*.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

docker-build:
	docker build -t healthcare-claims-glm:latest .

docker-run:
	docker-compose up -d

docker-down:
	docker-compose down

all: install data train evaluate test
