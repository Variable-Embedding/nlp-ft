PYTHON = python3
PIP = pip3

run: prep wikipedia-scraping model-our-data #model-given-data

model-our-data: prep
	$(PYTHON) main.py --config-file rnn_model_pipeline.yaml --topic=countries

#model-given-data: prep
#	$(PYTHON) main.py --config-file rnn_model_pipeline.yaml --topic=wiki

wikipedia-scraping: prep
	$(PYTHON) main.py --config-file wikipedia_scraping_pipeline.yaml --topic=countries

prep:
	mkdir -p logs tmp data

install:
	 $(PIP) install -r requirements.txt

clean:
	find . -type f -name \*.pyc -exec rm {} \;
	rm -rf dist *.egg-info .coverage .DS_Store logs tmp apicache-py3 *.lwp *.ctrl
