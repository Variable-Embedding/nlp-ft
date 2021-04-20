PYTHON = python3
PIP = pip3

prep:
	mkdir -p logs tmp data

install:
	 $(PIP) install -r requirements.txt

embeddings:
	$(PYTHON) main.py --config-file get_pre_trained_embeddings.yaml

benchmark:
	$(PYTHON) main.py --config-file get_benchmark_corpra.yaml

clean:
	find . -type f -name \*.pyc -exec rm {} \;
	rm -rf dist *.egg-info .coverage .DS_Store logs tmp apicache-py3 *.lwp *.ctrl
