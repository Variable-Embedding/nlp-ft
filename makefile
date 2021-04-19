PYTHON = python3
PIP = pip3

get-glove:
	$(PYTHON) main.py --config-file get_pre_trained_embeddings.yaml

prep:
	mkdir -p logs tmp data

install:
	 $(PIP) install -r requirements.txt

clean:
	find . -type f -name \*.pyc -exec rm {} \;
	rm -rf dist *.egg-info .coverage .DS_Store logs tmp apicache-py3 *.lwp *.ctrl
