test:
	python3 -m pytest tests -v --tb=no

lint:
	python3 -m pylint --errors-only ryan_adams

coverage:
	coverage run --source ryan_adams -m unittest discover -s tests
	coverage html
	open htmlcov/index.html

install:
	python3 -m pip install .[all]

docs: install
	$(MAKE) -C $(shell pwd)/docs html

.PHONY: docs
