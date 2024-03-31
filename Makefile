install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort **/*.py
	black **/*.py

all: install lint format