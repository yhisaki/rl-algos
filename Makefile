SHELL=/bin/bash

doc:
	cd doc && make html

format:
	isort rlrl/
	black rlrl/

clean:
	cd doc/ && make clean
	rm -rf wandb/

.PHONY: clean doc format
