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
	@echo ">>> Starting workflow from $(FILE) in conda env: $(CONDA_ENV) (running in background)"
	@mkdir -p $(LOGS_DIR)
	@nohup bash -c '\
		while IFS= read -r args || [ -n "$$args" ]; do \
			args_trimmed=$$(echo "$$args" | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$$//"); \
			if [[ -z "$$args_trimmed" || "$$args_trimmed" =~ ^# ]]; then continue; fi; \
			exp=$$(echo "$$args_trimmed" | awk "{print \$$1}"); \
			model=$$(echo "$$args_trimmed" | awk "{print \$$2}"); \
			version=$$(echo "$$args_trimmed" | awk "{print \$$NF}" | sed "s/version_dir=//"); \
			log_file="$(LOGS_DIR)/$${version}_$${model}_$${exp}.log"; \
			echo ">>> Running: $(BASE_CMD) $$args_trimmed (log: $$log_file)"; \
			conda run -n $(CONDA_ENV) bash -euo pipefail -c "$(BASE_CMD) $$args_trimmed" > "$$log_file" 2>&1; \
		done < $(FILE) \
	' > workflow_master.log 2>&1 &
	@echo ">>> Workflow launched in background (PID $$!). Logs in workflow_master.log + per-job logs."
