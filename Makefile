install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort **/*.py
	black **/*.py

lint:
	pylint --disable=R,C,W **/*.py

all: install lint format