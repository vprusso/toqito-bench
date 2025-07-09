# toqito-bench
Benchmarking suite for the toqito software package.

## Setup Environments General

### Python Environment
```bash
# Checks if the Python virtual environment exists at env/python-env.
make check-env-python

# Creates a Python 3.10 virtual environment in env/python-env, upgrades pip.
make setup-python

# Ensures the Python environment exists, creating it if missing.
make ensure-python

# Installs/updates pytest-benchmark in the virtual environment and runs setup/test.py.
make test-python

# Removes the Python virtual environment.
make clean-python

# Cleans and reinstalls the Python environment from scratch.
make reinstall-python
```
### Julia Environment

```bash
# Performs a fresh Julia installation, removing previous installations and setting up a new environment with BenchmarkTools.
make install-julia-fresh

# Checks for Julia installation; installs or updates as needed. Ensures BenchmarkTools is available in the project environment.
make setup-julia

# Runs Julia tests via setup/test.jl in the configured environment.
make test-julia

# Removes all Julia installations and environments.
make clean-julia
```

## Toqito Environment Setup & Benchmarking

### Setup Environment

```bash
# Checks if a Poetry environment for toqito exists at env/toqito-env.
make check-env-toqito

# Initializes a poetry environment for Toqito in env/toqito-env.
make setup-toqito

# Ensures the toqito poetry environment exists, creating it if missing.
make ensure-toqito

# Runs a setup test script (setup/test_toqito.py) inside the toqito environment.
make test-toqito-setup 

# Displays installed packages and environment info for toqito.
make toqito-info

#removes the Toqito Poetry environment.
make clean-toqito

# Reinstalls the toqito poetry environment from scratch.
make reinstall-toqito
```
### Benchmark

1. Runs all the tests in `scripts/benchmark_toqito.py` with `--benchmark-warmup=on` and `--benchmark-verbose`. JSON report is stored in `results/toqito/full`
```bash
make benchmark-full-toqito
```
2. Runs a simple benchmark with optional filtering and saving. Results are saved in `results/toqito/<filter>/<function>` with only required columns obtained from `--benchmark-columns=mean,median,stddev,ops`
As an example,
```bash
# runs benchmarks only for partial_trace function with varying only dim and displays the results but does not save them.
make benchmark-simple-toqito FILTER="TestPartialTraceBenchmarks" FUNCTION="test_bench__partial_trace__vary__dim" SAVE=false
```

3. Generates histogram visualizations from benchmark results or runs `make benchmark-simple-toqito` if none exist and then generate them. Histogram SVG files saved to `results/toqito/<filter>/<function>/images/`. As an example,

```bash
# If benchmarks for partial_trace function with varying only dim exist in isolation then constructs histogram through it else runs simple benchmarks for toqito with specified arguments and then construct it.
make benchmark-histogram-toqito FILTER="TestPartialTraceBenchmarks" FUNCTION="test_bench__partial_trace__vary__dim"
```

4. Runs benchmarks with memory profiling using `cProfile` (currently in development)
```bash
make test-toqito-benchmark-memory
```
## QuTIpy Environment Setup & Benchmarking

### Setup Environment

```bash
# Checks if a Poetry environment for qutipy exists at env/qutipy-env.
make check-env-qutipy

# Initializes a poetry environment for qutipy in env/qutipy-env.
make setup-qutipy

# Ensures the qutipy poetry environment exists, creating it if missing.
make ensure-qutipy

# Runs a setup test script (setup/test_qutipy.py) inside the qutipy environment.
make test-qutipy-setup 

#removes the qutipy Poetry environment.
make clean-qutipy

# Reinstalls the qutipy poetry environment from scratch.
make reinstall-qutipy
```

### Benchmark

1. Runs all the tests in `scripts/benchmark_qutipy.py` with `--benchmark-warmup=on` and `--benchmark-verbose`. JSON report is stored in `results/qutipy/full`
```bash
make benchmark-full-qutipy
```
2. Runs a simple benchmark with optional filtering and saving. Results are saved in `results/qutipy/<filter>/<function>` with only required columns obtained from `--benchmark-columns=mean,median,stddev,ops`
As an example,
```bash
# runs benchmarks only for partial_trace function with varying only dim and displays the results but does not save them.
make benchmark-simple-qutipy FILTER="TestPartialTraceBenchmarks" FUNCTION="test_bench__partial_trace__vary__dim" SAVE=false
```

3. Generates histogram visualizations from benchmark results or runs `make benchmark-simple-qutipy` if none exist and then generate them. Histogram SVG files saved to `results/qutipy/<filter>/<function>/images/`. As an example,

```bash
# If benchmarks for partial_trace function with varying only dim exist in isolation then constructs histogram through it else runs simple benchmarks for qutipy with specified arguments and then construct it.
make benchmark-histogram-qutipy FILTER="TestPartialTraceBenchmarks" FUNCTION="test_bench__partial_trace__vary__dim"
```