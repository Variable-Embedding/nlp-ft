PYTHON = python3
PIP = pip3
PYPY3 = pypy3

prep:
	mkdir -p logs tmp data

install:
	 $(PIP) install -r requirements.txt

install-pypy:
	 $(PYPY3) -m ensurepip
	 $(PYPY3) -mpip install -U pip wheel
	 $(PYPY3) -mpip install -r requirements.txt

embeddings:
	$(PYTHON) main.py --config-file get_pre_trained_embeddings.yaml

benchmark:
	$(PYTHON) main.py --config-file get_benchmark_corpra.yaml

benchmark2embeddings:
	$(PYTHON) main.py --config-file benchmark2embeddings.yaml

model:
	$(PYTHON) main.py --config-file run_model_pipeline.yaml

model-pypy:
	$(PYPY3) main.py --config-file run_model_pipeline.yaml

lm-experiment:
	$(PYTHON) main.py --config-file run_lm_experiment_pipeline.yaml

ft-experiment:
	$(PYTHON) main.py --config-file run_ft_experiment_pipeline.yaml

clean:
	find . -type f -name \*.pyc -exec rm {} \;
	rm -rf dist *.egg-info .coverage .DS_Store logs tmp apicache-py3 *.lwp *.ctrl
