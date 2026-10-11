.PHONY: setup lint test

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords')"

lint:
	ruff check .

test:
	pytest api/tests/
