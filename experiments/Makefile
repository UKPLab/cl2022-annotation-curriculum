download:
	python scripts/download_data.py
	git clone https://github.com/UKPLab/acl2016-convincing-arguments.git  acl2016-convincing-arguments data/external/acl2016-convincing-arguments

preprocess:
	python scripts/prepare_data.py

prepare: download preprocess
	python -m spacy download en_core_web_sm

shell:
	srun --pty bash -i

format:
	black -l 120 ca/
	black -l 120 scripts/
	black -l 120 userstudy/

	isort .

test:
	python -m pytest tests/

simulation:
	python slurm/submit_simulation.py