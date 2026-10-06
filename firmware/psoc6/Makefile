PYTHON ?= python3
PROJECT := mtb/build_workspace/psoc6_dual_core_ecg

.PHONY: all bootstrap getlibs build program check

all: build

bootstrap:
	$(PYTHON) mtb/bootstrap.py

check:
	$(PYTHON) mtb/bootstrap.py --check

getlibs: bootstrap
	$(MAKE) -C $(PROJECT) getlibs

build: bootstrap
	$(MAKE) -C $(PROJECT) build CONFIG=Release

program: build
	$(MAKE) -C $(PROJECT) program CONFIG=Release
