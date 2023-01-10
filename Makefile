
.DEFAULT: run

run:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01_preprocess.py

format:
	poetry run isort dev
	poetry run black dev
