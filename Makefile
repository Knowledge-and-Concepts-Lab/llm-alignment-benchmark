SHELL := /bin/bash

.PHONY: workflow

WORKFLOW ?= commands.txt
FILE := workflows/$(WORKFLOW)


CONDA_ENV := alignment_benchmark_env

BASE_CMD ?= python3 script/exp_battery.py

LOGS_DIR := logs

workflow:
	@if [ ! -f $(FILE) ]; then \
		echo "ERROR: Workflow file $(FILE) not found."; \
		exit 1; \
	fi
	@echo ">>> Running workflow from $(FILE) in conda env: $(CONDA_ENV)"
	@mkdir -p $(LOGS_DIR)
	@i=1; \
	while IFS= read -r args || [ -n "$$args" ]; do \
		args_trimmed=$$(echo "$$args" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$$//'); \
		if [[ -z "$$args_trimmed" || "$$args_trimmed" =~ ^# ]]; then continue; fi; \
		log_file="$(LOGS_DIR)/$${i}.log"; \
		echo ">>> Running: $(BASE_CMD) $$args_trimmed (log: $$log_file)"; \
		nohup conda run -n $(CONDA_ENV) bash -euo pipefail -c '$(BASE_CMD) '"$$args_trimmed" > "$$log_file" 2>&1 & \
		wait $$!; \
		i=$$((i+1)); \
	done < $(FILE)
