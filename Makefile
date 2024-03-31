install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort .
	black .

lint:
	pylint --disable=R,C,W **/*.py

all: install format lint
