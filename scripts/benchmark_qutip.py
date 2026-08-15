"""Benchmarks for QuTiP, for comparison against toqito.

QuTiP is a different package from QuTiPy, which `benchmark_qutipy.py` covers.
QuTiPy is Sumeet Khatri's quantum-information package; QuTiP (Johansson, Nation,
Nori) is the much more widely used simulation framework. Both are worth
comparing against, and their coverage of toqito's API differs, so they get
separate modules.

Parameter values mirror `benchmark_toqito.py` exactly so the two sets of numbers
can be read side by side.

A note on what is timed. QuTiP operates on `Qobj`, which carries subsystem
dimension metadata, whereas toqito takes bare NumPy arrays. Wrapping an array in
a `Qobj` inside the timed region would measure construction overhead that has no
counterpart on the toqito side and would make the comparison meaningless. Every
benchmark here therefore builds its `Qobj` in setup and times only the operation
itself. Where QuTiP exposes an operation as a method rather than a function
(`ptrace`, `permute`, `norm`), the bound method is what gets timed.
"""

import numpy as np
import pytest
import qutip


class TestPartialTraceBenchmarks:
    """Benchmarks for QuTiP's `Qobj.ptrace`, the counterpart to `toqito.channels.partial_trace`."""

    @pytest.mark.parametrize("matrix_size", [4, 16, 64, 256], ids=lambda x: str(x))
    def test_bench__partial_trace__vary__input_mat(self, benchmark, matrix_size):
        """Benchmark `ptrace` with varying input matrix sizes.

        Sizes match the toqito benchmark of the same name.

        Fixed Parameters:
            - subsystem traced out: the second, matching toqito's default behaviour.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (n) of the n x n input matrix.
        """
        n = int(round(np.sqrt(matrix_size)))
        mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        rho = qutip.Qobj(mat, dims=[[n, n], [n, n]])

        result = benchmark(rho.ptrace, 0)

        assert result.shape[0] <= matrix_size


class TestPartialTransposeBenchmarks:
    """Benchmarks for `qutip.partial_transpose`, the counterpart to `toqito.channels.partial_transpose`."""

    @pytest.mark.parametrize("matrix_size", [4, 16, 64], ids=lambda x: str(x))
    def test_bench__partial_transpose__vary__input_mat(self, benchmark, matrix_size):
        """Benchmark `partial_transpose` with varying input matrix sizes.

        Fixed Parameters:
            - `mask`: transpose the first subsystem only.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (n) of the n x n input matrix.
        """
        n = int(round(np.sqrt(matrix_size)))
        mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        rho = qutip.Qobj(mat, dims=[[n, n], [n, n]])

        result = benchmark(qutip.partial_transpose, rho, [1, 0])

        assert result.shape[0] == matrix_size


class TestTraceDistanceBenchmarks:
    """Benchmarks for `qutip.tracedist`, the counterpart to `toqito.state_metrics.trace_distance`."""

    @pytest.mark.parametrize("dim", [4, 16, 64], ids=lambda x: str(x))
    def test_bench__trace_distance__vary__dim(self, benchmark, dim):
        """Benchmark the trace distance between two random density matrices.

        Fixed Parameters:
            - both inputs are Haar-random density matrices of the same dimension.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of each density matrix.
        """
        rho = qutip.rand_dm(dim)
        sigma = qutip.rand_dm(dim)

        result = benchmark(qutip.tracedist, rho, sigma)

        assert result >= -1e-9


class TestTraceNormBenchmarks:
    """Benchmarks for `Qobj.norm("tr")`, the counterpart to `toqito.matrix_props.trace_norm`."""

    @pytest.mark.parametrize("dim", [4, 16, 64], ids=lambda x: str(x))
    def test_bench__trace_norm__vary__dim(self, benchmark, dim):
        """Benchmark the trace norm of a random density matrix.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the density matrix.
        """
        rho = qutip.rand_dm(dim)

        result = benchmark(rho.norm, "tr")

        assert result >= -1e-9


class TestRandomDensityMatrixBenchmarks:
    """Benchmarks for `qutip.rand_dm`, the counterpart to `toqito.rand.random_density_matrix`."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256], ids=lambda x: str(x))
    def test_bench__random_density_matrix__vary__dim(self, benchmark, dim):
        """Benchmark random density matrix generation across dimensions.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the generated density matrix.
        """
        result = benchmark(qutip.rand_dm, dim)

        assert result.shape[0] == dim


class TestRandomUnitaryBenchmarks:
    """Benchmarks for `qutip.rand_unitary`, the counterpart to `toqito.rand.random_unitary`."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256], ids=lambda x: str(x))
    def test_bench__random_unitary__vary__dim(self, benchmark, dim):
        """Benchmark random unitary generation across dimensions.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the generated unitary.
        """
        result = benchmark(qutip.rand_unitary, dim)

        assert result.shape[0] == dim


class TestVonNeumannEntropyBenchmarks:
    """Benchmarks for `qutip.entropy_vn`, the counterpart to `toqito.state_props.von_neumann_entropy`."""

    @pytest.mark.parametrize("dim", [4, 16, 64], ids=lambda x: str(x))
    def test_bench__von_neumann_entropy__vary__dim(self, benchmark, dim):
        """Benchmark the von Neumann entropy of a random density matrix.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the density matrix.
        """
        rho = qutip.rand_dm(dim)

        result = benchmark(qutip.entropy_vn, rho)

        assert np.isfinite(result)


class TestPermuteSystemsBenchmarks:
    """Benchmarks for `Qobj.permute`, the counterpart to `toqito.perms.permute_systems`."""

    @pytest.mark.parametrize("num_qubits", [2, 4, 6], ids=lambda x: str(x))
    def test_bench__permute_systems__vary__num_subsystems(self, benchmark, num_qubits):
        """Benchmark permuting subsystems as the number of qubits grows.

        Fixed Parameters:
            - permutation: reverse the subsystem order.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            num_qubits (int): Number of qubit subsystems.
        """
        rho = qutip.rand_dm([2] * num_qubits)
        order = list(reversed(range(num_qubits)))

        result = benchmark(rho.permute, order)

        assert result.shape[0] == 2**num_qubits


class TestBasisBenchmarks:
    """Benchmarks for `qutip.basis`, the counterpart to `toqito.states.basis`."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256], ids=lambda x: str(x))
    def test_bench__basis__vary__dim(self, benchmark, dim):
        """Benchmark standard basis vector construction across dimensions.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the space.
        """
        result = benchmark(qutip.basis, dim, 0)

        assert result.shape[0] == dim


class TestToDensityMatrixBenchmarks:
    """Benchmarks for `qutip.ket2dm`, the counterpart to `toqito.matrix_ops.to_density_matrix`."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256], ids=lambda x: str(x))
    def test_bench__to_density_matrix__vary__dim(self, benchmark, dim):
        """Benchmark converting a ket to a density matrix across dimensions.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of the ket.
        """
        ket = qutip.basis(dim, 0)

        result = benchmark(qutip.ket2dm, ket)

        assert result.shape[0] == dim
