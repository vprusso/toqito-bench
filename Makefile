PYTHON_VERSION := 3.10
BENCHMARK_DEPS := pytest-benchmark
VENV_DIR := env/python-env

# Check if virtual environment exists.
check-env-python:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
		exit 0; \
	else \
		echo "No existing virtual environment found"; \
		exit 1; \
	fi

# Setup Python virtual environment.
setup-python: check-env-python
	@if [ $$? -eq 1 ]; then \
		echo "Creating Python $(PYTHON_VERSION) virtual environment..."; \
		mkdir -p env; \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
		echo "Upgrading pip..."; \
		$(VENV_DIR)/bin/python -m pip install --upgrade pip; \
		echo "Virtual environment ready"; \
	fi

# Verify Python environment existence (create if missing).
ensure-python:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(MAKE) setup-python; \
	fi

# Execute Python benchmarks
test-python: ensure-python
	@echo "Installing/updating test dependencies..."
	@$(VENV_DIR)/bin/python -m pip install $(BENCHMARK_DEPS)
	@echo "Running tests..."
	@$(VENV_DIR)/bin/python setup/test.py
	@echo "Tests completed!"

# Remove Python virtual environment.
clean-python:
	@echo "Cleaning up Python virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Environment removed ..."

# Reinstall Python environment.
reinstall-python: clean-python setup-python
	@echo "Python $(PYTHON_VERSION) reinstallation complete"


###ENVIRONMENT SETUP FOR  TOQITO#####
TOQITO_ENV_DIR := env/toqito-env
PYPROJECT_FILE_TOQITO := $(TOQITO_ENV_DIR)/pyproject.toml

check-env-toqito:
	@if [ -f "$(PYPROJECT_FILE_TOQITO)" ]; then \
		echo "Poetry toqito environment already exists at $(TOQITO_ENV_DIR)"; \
		exit 0; \
	else \
		echo "No existing toqito Poetry environment found, run setup-toqito"; \
		exit 1; \
	fi

setup-toqito:
	@if [ -d "$(TOQITO_ENV_DIR)" ]; then \
		echo "Environment directory $(TOQITO_ENV_DIR) already exists."; \
		echo "Run 'make clean-toqito' to remove it before setting up again."; \
		exit 1; \
	fi
	@echo "Creating dedicated toqito Poetry environment..."
	@mkdir -p $(TOQITO_ENV_DIR)
	@cd $(TOQITO_ENV_DIR) && poetry init --no-interaction \
		--name "toqito-benchmarks" \
		--python "^3.10"
	@echo 'package-mode = false' >> $(TOQITO_ENV_DIR)/pyproject.toml
	@echo "Adding toqito and benchmark dependencies..."
	@cd $(TOQITO_ENV_DIR) && poetry add toqito pytest-benchmark --group dev pytest-memray sympy pygal
	@echo "Installing dependencies..."
	@echo "toqito poetry environment ready at $(TOQITO_ENV_DIR)"


ensure-toqito:
	@if [ ! -f "$(PYPROJECT_FILE_TOQITO)" ]; then \
		$(MAKE) setup-toqito; \
	fi

test-toqito-setup: ensure-toqito
	@echo "Running toqito setup test with poetry..."
	@cd $(TOQITO_ENV_DIR) && poetry run python ../../setup/test_toqito.py
	@echo "toqito setup test completed!"

BENCHMARK_REPORTS := reports
BENCHMARK_DIR := scripts
BENCHMARK_FILE_TOQITO := benchmark_toqito.py
BENCHMARK_STORAGE := results

# MAY add functionality:
# --benchmark-cprofile=cumulative

benchmark-full-toqito: ensure-toqito
	@echo "Running detailed benchmarks for toqito..."
	@mkdir -p $(BENCHMARK_REPORTS)/toqito
	@mkdir -p $(BENCHMARK_STORAGE)/toqito/full
	@cd $(TOQITO_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_TOQITO) \
		$(if $(FILTER),-k "$(FILTER)") \
		$(if $(FUNCTION),-m "$(FUNCTION)") \
		--benchmark-warmup=on \
		--benchmark-sort=name \
		--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds \
		--benchmark-save=detailed_$(shell date +%Y_%m_%d__%H_%M_%S) \
		--benchmark-storage=$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/full \
		--benchmark-verbose \
		-v --tb=long
	@echo "Detailed benchmarks for toqito completed !"

benchmark-simple-toqito: ensure-toqito
	@echo "Starting simple benchmark run for toqito..."
	@echo " Benchmark file: $(BENCHMARK_FILE_TOQITO)"
	@echo " Filter applied: $(if $(FILTER),$(FILTER),none)"
	@echo " Function applied: $(if $(FUNCTION),$(FUNCTION),none)"
	@echo " Storage: $(if $(SAVE),$(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION),not saving results)"

	$(if $(SAVE),@mkdir -p "$(shell pwd)/$(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION)",)
	@cd $(TOQITO_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_TOQITO) \
		$(if $(FILTER),-k "$(FILTER)",) \
		$(if $(FUNCTION),-k "$(FUNCTION)",) \
		--benchmark-sort=name \
		--benchmark-columns=mean,median,stddev,ops\
		$(if $(SAVE),--benchmark-save=simple_$(shell date +%Y_%m_%d__%H_%M_%S),) \
		$(if $(SAVE),--benchmark-storage="$(shell pwd)/$(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION)",) \
		--tb=short
	@echo "Simple benchmarks completed successfully."

benchmark-histogram-toqito: ensure-toqito
	@echo " Checking for existing benchmark results in toqito..."

	@mkdir -p "$(shell pwd)/$(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION)/images"

	@STORAGE_DIR="$(shell pwd)/$(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION)" && \
	echo " Looking for existing results in: $$STORAGE_DIR" && \
	LATEST_JSON=$$(find "$$STORAGE_DIR"/Linux-* -name "*.json" -type f 2>/dev/null | sort | tail -n 1) && \
	if [ -n "$$LATEST_JSON" ]; then \
		BASE_JSON=$$(basename "$$LATEST_JSON") && \
		echo " Found existing result: $$BASE_JSON"; \
		HISTOGRAM_PREFIX="$$STORAGE_DIR/images/histogram_$$(date +%Y_%m_%d__%H_%M_%S)__from_$${BASE_JSON%.json}" && \
		echo " Generating histogram: $$HISTOGRAM_PREFIX.svg"; \
		cd $(TOQITO_ENV_DIR) && poetry run py.test-benchmark compare \
			"$$LATEST_JSON" \
			--histogram "$$HISTOGRAM_PREFIX" \
			--sort name \
			--columns mean,max,min,stddev; \
	else \
		SAVE_NAME="simple_$$(date +%Y_%m_%d__%H_%M_%S)" && \
		BASE_JSON="$$SAVE_NAME.json" && \
		HISTOGRAM_PREFIX="$$STORAGE_DIR/images/histogram_$$(date +%Y_%m_%d__%H_%M_%S)__from_$${BASE_JSON%.json}" && \
		echo "  No existing results found. Running benchmark-simple..."; \
		echo " Will save as: $$BASE_JSON"; \
		echo " Histogram will be saved as: $$HISTOGRAM_PREFIX-<test_name>.svg"; \
		cd $(TOQITO_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_TOQITO) \
			$(if $(FILTER),-k "$(FILTER)",) \
			$(if $(FUNCTION),-k "$(FUNCTION)",) \
			--benchmark-sort=name \
			--benchmark-columns=mean,max,min,stddev \
			--benchmark-save=$$SAVE_NAME \
			--benchmark-storage="$$STORAGE_DIR" \
			--benchmark-histogram="$$HISTOGRAM_PREFIX" \
			--tb=short; \
	fi

	@echo " Histogram generation completed!"
	@echo " Histogram(s) saved to: $(BENCHMARK_STORAGE)/toqito/$(FILTER)/$(FUNCTION)/images/"


test-toqito-benchmark-memory: ensure-toqito
	@echo "Running toqito benchmarks with Poetry..."
ifdef FILTER
	@cd $(TOQITO_ENV_DIR) && poetry run pytest ../../scripts/benchmark_toqito.py -k "$(FILTER)" --benchmark-sort=name --benchmark-columns=min,mean,max --benchmark-save=toqito --memray --memray-bin-path=./memray-results
else
	@cd $(TOQITO_ENV_DIR) && poetry run pytest ../../scripts/benchmark_toqito.py --benchmark-sort=name --benchmark-columns=min,mean,max --benchmark-save=toqito --memray --memray-bin-path=./memray-results
endif
	@echo "toqito benchmarks completed!"
	@echo "Memory profiling results saved to ./memray-results/"


toqito-info: ensure-toqito
	@echo "Toqito environment information:"
	@cd $(TOQITO_ENV_DIR) && poetry show
	@cd $(TOQITO_ENV_DIR) && poetry env info

clean-toqito:
	@echo "Cleaning up toqito Poetry environment..."
	@if [ -d "$(TOQITO_ENV_DIR)" ]; then \
		cd $(TOQITO_ENV_DIR) && poetry env remove --all 2>/dev/null || true; \
	fi
	@rm -rf $(TOQITO_ENV_DIR)
	@echo "toqito poetry environment removed"

reinstall-toqito: clean-toqito setup-toqito
	@echo "toqito poetry environment reinstallation complete"

##############SETUP FOR QUTIPY####################

QUTIPY_ENV_DIR := env/qutipy-env
PYPROJECT_FILE_QUTIPY := $(QUTIPY_ENV_DIR)/pyproject.toml

check-env-qutipy:
	@if [ -f "$(PYPROJECT_FILE_QUTIPY)" ]; then \
		echo "Poetry QuTIpy environment already exists at $(QUTIPY_ENV_DIR)"; \
		exit 0; \
	else \
		echo "No existing QuTIpy Poetry environment found, run setup-qutipy"; \
		exit 1; \
	fi

setup-qutipy:
	@if [ -d "$(QUTIPY_ENV_DIR)" ]; then \
		echo "Environment directory $(QUTIPY_ENV_DIR) already exists."; \
		echo "Run 'make clean-qutipy' to remove it before setting up again."; \
		exit 1; \
	fi
	@echo "Creating dedicated QuTIpy Poetry environment..."
	@mkdir -p $(QUTIPY_ENV_DIR)
	@cd $(QUTIPY_ENV_DIR) && poetry init --no-interaction \
		--name "qutipy-benchmarks" \
		--python ">=3.11,<4.0"
	@echo 'package-mode = false' >> $(QUTIPY_ENV_DIR)/pyproject.toml
	@echo "Adding QuTIpy and benchmark dependencies..."
	@cd $(QUTIPY_ENV_DIR) && poetry add git+https://github.com/sumeetkhatri/QuTIpy pytest-benchmark numpy scipy cvxpy sympy pygal
	@echo "Installing dependencies..."
	@echo "QuTIpy poetry environment ready at $(QUTIPY_ENV_DIR)"


ensure-qutipy:
	@if [ ! -f "$(PYPROJECT_FILE_QUTIPY)" ]; then \
		$(MAKE) setup-qutipy; \
	fi

test-qutipy-setup: ensure-qutipy
	@echo "Running QuTIpy setup test with poetry..."
	@cd $(QUTIPY_ENV_DIR) && poetry run python ../../setup/test_qutipy.py
	@echo "QuTIpy setup test completed!"


BENCHMARK_FILE_QUTIPY := benchmark_qutipy.py

benchmark-full-qutipy: ensure-qutipy
	@echo "Running detailed benchmarks for qutipy..."
	@mkdir -p $(BENCHMARK_REPORTS)/qutipy
	@mkdir -p $(BENCHMARK_STORAGE)/qutipy/full
	@cd $(QUTIPY_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_QUTIPY) \
		$(if $(FILTER), -k "$(FILTER)")\
		$(if $(FUNCTION),-m "$(FUNCTION)") \
		--benchmark-warmup=on \
		--benchmark-sort=name \
		--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds \
		--benchmark-save=detailed_$(shell date +%Y_%m_%d__%H_%M_%S) \
		--benchmark-storage=$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/full \
		--benchmark-verbose \
		-v --tb=long
	@echo "Detailed benchmarks for qutipy completed !"


benchmark-simple-qutipy: ensure-qutipy
	@echo "Starting simple benchmark run for qutipy..."
	@echo " Benchmark file: $(BENCHMARK_FILE_QUTIPY)"
	@echo " Filter applied: $(if $(FILTER),$(FILTER),none)"
	@echo " Function applied: $(if $(FUNCTION),$(FUNCTION),none)"
	@echo " Storage: $(if $(SAVE),$(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION),not saving results)"

	$(if $(SAVE),@mkdir -p "$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION)",)
	@cd $(QUTIPY_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_QUTIPY) \
		$(if $(FILTER),-k "$(FILTER)",) \
		$(if $(FUNCTION),-k "$(FUNCTION)",) \
		--benchmark-sort=name \
		--benchmark-columns=mean,median,stddev,ops  \
		$(if $(SAVE),--benchmark-save=simple_$(shell date +%Y_%m_%d__%H_%M_%S),) \
		$(if $(SAVE),--benchmark-storage="$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION)",) \
		--tb=short
	@echo "Simple benchmarks completed for qutipy successfully."

benchmark-histogram-qutipy: ensure-qutipy
	@echo " Checking for existing benchmark results in qutipy..."

	@mkdir -p "$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION)/images"

	@STORAGE_DIR="$(shell pwd)/$(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION)" && \
	echo " Looking for existing results in: $$STORAGE_DIR" && \
	LATEST_JSON=$$(find "$$STORAGE_DIR"/Linux-* -name "*.json" -type f 2>/dev/null | sort | tail -n 1) && \
	if [ -n "$$LATEST_JSON" ]; then \
		BASE_JSON=$$(basename "$$LATEST_JSON") && \
		echo " Found existing result: $$BASE_JSON"; \
		HISTOGRAM_PREFIX="$$STORAGE_DIR/images/histogram_$$(date +%Y_%m_%d__%H_%M_%S)__from_$${BASE_JSON%.json}" && \
		echo " Generating histogram: $$HISTOGRAM_PREFIX.svg"; \
		cd $(QUTIPY_ENV_DIR) && poetry run py.test-benchmark compare \
			"$$LATEST_JSON" \
			--histogram "$$HISTOGRAM_PREFIX" \
			--sort name \
			--columns mean,max,min,stddev; \
	else \
		SAVE_NAME="simple_$$(date +%Y_%m_%d__%H_%M_%S)" && \
		BASE_JSON="$$SAVE_NAME.json" && \
		HISTOGRAM_PREFIX="$$STORAGE_DIR/images/histogram_$$(date +%Y_%m_%d__%H_%M_%S)__from_$${BASE_JSON%.json}" && \
		echo "  No existing results found for qutipy. Running benchmark-simple..."; \
		echo " Will save as: $$BASE_JSON"; \
		echo " Histogram will be saved as: $$HISTOGRAM_PREFIX-<test_name>.svg"; \
		cd $(QUTIPY_ENV_DIR) && poetry run pytest ../../$(BENCHMARK_DIR)/$(BENCHMARK_FILE_QUTIPY) \
			$(if $(FILTER),-k "$(FILTER)",) \
			$(if $(FUNCTION),-k "$(FUNCTION)",) \
			--benchmark-sort=name \
			--benchmark-columns=mean,max,min,stddev \
			--benchmark-save=$$SAVE_NAME \
			--benchmark-storage="$$STORAGE_DIR" \
			--benchmark-histogram="$$HISTOGRAM_PREFIX" \
			--tb=short; \
	fi

	@echo " Histogram generation completed!"
	@echo " Histogram(s) saved to: $(BENCHMARK_STORAGE)/qutipy/$(FILTER)/$(FUNCTION)/images/"



clean-qutipy:
	@echo "Cleaning up QuTIpy Poetry environment..."
	@if [ -d "$(QUTIPY_ENV_DIR)" ]; then \
		cd $(QUTIPY_ENV_DIR) && poetry env remove --all 2>/dev/null || true; \
	fi
	@rm -rf $(QUTIPY_ENV_DIR)
	@echo "QuTIpy poetry environment removed"

reinstall-qutipy: clean-qutipy setup-qutipy
	@echo "qutipy poetry environment reinstallation complete"

##########BASIC JULIA ENVIRONMENT SETUP############
JULIA_VERSION := 1.10
JULIA_DIR := $(HOME)/.julia
JULIAUP_DIR := $(HOME)/.juliaup
JULIA_ENV := env/julia-env
JULIAUP_BIN := $(JULIAUP_DIR)/bin

# Fresh Julia installation proceedure.
install-julia-fresh:
	@echo "Performing Julia cleanup for fresh installation ... "
	@rm -rf "$(JULIA_DIR)" "(JULIAUP_DIR)" "$(JULIA_ENV)"


	@echo "Installing Julia $(JULIA_VERSION) ..."
	@curl -fsSL https://install.julialang.org | sh -s -- --yes || (echo "❌ Juliaup installation failed"; exit 1)
	

	@echo "Verifying juliaup installation ... "
	@if [ -f "$(JULIAUP_BIN)/juliaup" ]; then \
		echo "Setting Julia $(JULIA_VERSION) as default ..."; \
		"$(JULIAUP_BIN)/juliaup" add $(JULIA_VERSION); \
		"$(JULIAUP_BIN)/juliaup" default $(JULIA_VERSION); \
	else\
		echo "Juliaup binary not found !!!"; \
		exit 1; \
	fi


	@echo "Creating project environment ... "
	@mkdir -p $(JULIA_ENV)
	@"$(JULIAUP_BIN)/julia" --project=$(JULIA_ENV) -e "using Pkg; Pkg.add(\"BenchmarkTools\"); Pkg.instantiate()"
	@echo "Julia $(JULIA_VERSION) setup complete in $(JULIA_ENV)."


# Julia environment setup.
setup-julia:
	@echo "Checking for existing Julia $(JULIA_VERSION)..."
	@if command -v julia >/dev/null 2>&1; then \
		CURRENT_VERSION=$$(julia --version 2>/dev/null | sed -n 's/julia version \([0-9]\+\.[0-9]\+\).*/\1/p'); \
		if [ "$$CURRENT_VERSION" = "$(JULIA_VERSION)" ]; then \
			echo "Julia $(JULIA_VERSION) already installed ($$CURRENT_VERSION)"; \
			echo "Checking/updating project environment..."; \
			mkdir -p $(JULIA_ENV); \
			julia --project=$(JULIA_ENV) -e '\
				using Pkg; \
				if !isfile("$(JULIA_ENV)/Project.toml") || !haskey(Pkg.project().dependencies, "BenchmarkTools"); \
					println("Setting up BenchmarkTools..."); \
					Pkg.add("BenchmarkTools"); \
				else \
					println("BenchmarkTools already configured"); \
				end; \
				Pkg.instantiate(); \
				println("Project status:"); \
				Pkg.status()'; \
			echo "Julia environment ready at $(JULIA_ENV)"; \
		else \
			echo "Julia $$CURRENT_VERSION found, but need $(JULIA_VERSION)"; \
			echo "Installing Julia $(JULIA_VERSION)..."; \
			$(MAKE) install-julia-fresh; \
		fi; \
	else \
		echo "Julia not found, performing fresh installation..."; \
		$(MAKE) install-julia-fresh; \
	fi

# Execute Julia tests.
test-julia: setup-julia
	@echo "Running Julia tests..."
	@"$(JULIAUP_BIN)/julia" --project=$(JULIA_ENV) setup/test.jl

# Clean Julia installations.
clean-julia:
	@echo "Removing all Julia installations..."
	@rm -rf "$(JULIA_DIR)" "$(JULIAUP_DIR)" "$(JULIA_ENV)"
	@echo "Everything removed. Fresh start available."