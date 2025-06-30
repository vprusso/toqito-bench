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
	@$(VENV_DIR)/bin/python benchmarks/test.py
	@echo "Tests completed!"

# Remove Python virtual environment.
clean-python:
	@echo "Cleaning up Python virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Environment removed ..."

# Reinstall Python environment.
reinstall-python: clean-python setup-python
	@echo "Python $(PYTHON_VERSION) reinstallation complete"


JULIA_VERSION := 1.10
JULIA_DIR := $(HOME)/.julia
JULIAUP_DIR := $(HOME)/.juliaup
JULIA_ENV := env/julia-env
JULIAUP_BIN := $(JULIAUP_DIR)/bin

# Fresh Julia installation procedure.
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
	@"$(JULIAUP_BIN)/julia" --project=$(JULIA_ENV) benchmarks/test.jl

# Clean Julia installations.
clean-julia:
	@echo "Removing all Julia installations..."
	@rm -rf "$(JULIA_DIR)" "$(JULIAUP_DIR)" "$(JULIA_ENV)"
	@echo "Everything removed. Fresh start available."