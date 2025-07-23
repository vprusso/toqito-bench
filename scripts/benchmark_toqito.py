import pytest
import numpy as np
import scipy as sp
import itertools
from cvxpy.expressions.variable import Variable

from toqito.channels import partial_trace


from toqito.matrix_ops import to_density_matrix


from toqito.rand import random_density_matrix
from toqito.rand import random_unitary
from toqito.rand import random_psd_operator

from toqito.state_metrics import trace_distance

from toqito.matrix_props import trace_norm
from toqito.matrix_props import is_positive_semidefinite

from toqito.state_props import log_negativity
from toqito.state_props import von_neumann_entropy

from toqito.channel_ops import natural_representation
from toqito.channel_ops import kraus_to_choi
from toqito.channel_ops import apply_channel

from toqito.channels import amplitude_damping
from toqito.channels import bitflip
from toqito.channels import dephasing
from toqito.channels import partial_transpose


from toqito.perms import swap
from toqito.perms import swap_operator
from toqito.perms import permute_systems
from toqito.perms import permutation_operator
from toqito.matrix_ops import vec

from toqito.states import basis

from toqito.matrices import pauli

class TestPartialTraceBenchmarks:
    """Benchmarks for the `toqito.channels.partial_trace` function."""

    @pytest.mark.parametrize(
        "matrix_size",
        [4, 16, 64, 256],
        ids = lambda x: str(x),
    )
    def test_bench__partial_trace__vary__input_mat(self, benchmark, matrix_size):
        """Benchmark `partial_trace` with varying input matrix sizes.

        Fixed Parameters:
            - `sys`: Set to `None` to use default behaviour of trace of the second subsystem.
            - `dim`: Set to `None` to use default behaviour, which infers the subsystem dimensions as a square system of size `sqrt(len(input_mat))`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (n) of the n x n input matrix.
        """
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        
        result = benchmark(partial_trace, input_mat = input_mat, sys = None, dim = None)

        assert result.shape[0] <= matrix_size
    
    @pytest.mark.parametrize(
        "sys",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
        ],
        ids = lambda x: str(x),
    )
    def test_bench__partial_trace__vary__sys(self, benchmark, sys):
        """Benchmark `partial_trace` by tracing out different subsystems.

        Fixed Parameters:
            - `input_mat`: Generated with a constant matrix size of `16 x 16`.
            - `dim`: Dynmaically set to be compatible with `sys` value.
        
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sys (list[int]) : system to take trace of.
        """

        # Generate a random matrix of size 16x16.
        matrix_size = 16
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        if sys == [0, 2]:
            # Assume input_mat is composed of 4 systems with dim = 2 each.
            dim = [2, 2, 2, 2]
        elif sys == [0, 1]:
            # Assume input_mat is composed of 2 systems with dim = 4 each.
            dim = [4, 4]
        else:
            # Assume default behaviour.
            dim = None
        
        result = benchmark(partial_trace, input_mat = input_mat, sys = sys, dim = dim)

        assert result is not None
    
    @pytest.mark.parametrize(
        "dim",
        [
            None,
            [2, 2],
            [2, 2, 2, 2],
            [3, 3],
            [4, 4],
        ],
        ids = lambda x: str(x),
    )
    def test_bench__parital_trace__vary__dim(self, benchmark, dim):
        """Benchmark `partial_trace` by varying subsystem dimensions (`dim`).

        Fixed Parameters:
            - `input_mat`: Random matrix generated with size set as product of the dimension in `dim`
            - `sys`: set to `None` to use default behaviour of trace of the second subsystem.
        
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (list[int] | None): A list specifying the dimension of each subsystem.
        """
        if dim is None:
            # Use a fixed matrix size of 16 for default dim.
            matrix_size = 16
        else:
            matrix_size = int(np.prod(dim))
        
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        result = benchmark(partial_trace, input_mat = input_mat, sys = None, dim = dim)

        assert result is not None
    
class TestRandomDensityMatrixBenchmarks:
    """Benchmarks for the `toqito.rand.random_density_matrix` function."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024], ids = lambda x: str(x))
    def test_bench__random_density_matrix__vary__dim(self, benchmark, dim):
        """Benchmark `random_density_matrix` with varying output sizes.
        
        Fixed Parameters:
            - `is_real`: Set to `False` to generate complex-valued matrices.
            - `k_param`: Set to `None` to generate full-rank density matrices.
            - `distance_metric`: Set to `"haar"` for the standard Haar measure.
            - `seed`: Set to `None` so that a random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the output density matrix.
        """
        result = benchmark(
            random_density_matrix,
            dim = dim,
            is_real = False,
            k_param = None,
            distance_metric = "haar",
            seed = None,
        )
        
        assert result.shape == (dim, dim)
    
    @pytest.mark.parametrize(
        "dim, k_param",
        [
            (4, 1),
            (4, 2),
            (16, 1),
            (16, 8),
            (64, 1),
            (64, 32),
            (256, 1),
            (256, 128),
            (1024, 1),
            (1024, 512)
        ],
        ids=lambda x: f"{x}"
    )
    def test_bench__random_density_matrix__vary__dim_kparam(self, benchmark, dim, k_param):
        """Benchmark `random_density_matrix` with varying dimensions and ranks.

        Fixed Parameters:
             - `is_real`: Set to `False` to generate complex-valued matrices.
             - `distance_metric`: Set to `"haar"` for the standard Haar measure.
             - `seed`: Set to `None` so that a random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the output density matrix.
            k_param (int): The rank of the density matrix.

        """
        result = benchmark(
            random_density_matrix,
            dim = dim,
            is_real = False,
            k_param = None,
            distance_metric = "haar",
            seed = None,
        )

        assert result.shape == (dim, dim) 

    @pytest.mark.parametrize("is_real", [True, False], ids = lambda x:str(x))
    def test_bench__random_density_matrix__param__is_real(self, benchmark, is_real):
        """Benchmark `random_density_matrix` for real and complex outputs.

        Fixed Parameters:
            - `dim`: A constant dimension of 64 is used for the matrix.
            - `k_param`: Set to `None` to generate full-rank density matrices.
            - `distance_metric`: Set to `"haar"` for the standard Haar measure.
            - `seed`: Set to `None` so that a random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            is_real (bool): If `True`, generates a real-valued matrix; otherwise, complex.
        """
        dim = 64 # select a constant dimension.

        result = benchmark(
            random_density_matrix,
            dim = dim,
            is_real = is_real,
            k_param = None,
            distance_metric = "haar",
            seed = None,
        )

        assert result.shape == (dim, dim)
    
    @pytest.mark.parametrize("distance_metric", ["haar", "bures"], ids = lambda x: str(x))
    def test_bench__random_density_matrix__param__distance_metric(self, benchmark, distance_metric):
        """Benchmark `random_density_matrix` with different distance metrics.

        Fixed Parameters:
            - `dim`: A constant dimension of 64 is used for the matrix.
            - `is_real`: Set to `False` to generate complex-valued matrices.
            - `k_param`: Set to `None` to generate full-rank density matrices.
            - `seed`: Set to `None` so that a random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            distance_metric (str): The distance metric to use for generation.
        """
        dim = 64 # select a constant dimension.

        result = benchmark(
            random_density_matrix,
            dim = dim,
            is_real = False,
            k_param = None,
            distance_metric = distance_metric,
            seed = None,
        )
        

        assert result.shape == (dim, dim)

class TestRandomUnitaryBenchmarks:
    """Benchmarks for the `toqito.rand.random_unitary` function."""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024], ids=lambda x: str(x))
    def test_bench__random_unitary__vary__dim(self, benchmark, dim):
        """Benchmark `random_unitary` with varying matrix dimensions.

        Fixed Parameters:
            - `is_real`: Set to `False` to generate complex-valued unitary matrices.
            - `seed`: Set to `None` so a new random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the unitary matrix to generate.
        """
        result = benchmark(random_unitary, dim=dim, is_real=False, seed=None)

        assert result.shape == (dim, dim)

    @pytest.mark.parametrize("is_real", [True, False], ids=lambda x: str(x))
    def test_bench__random_unitary__vary__is_real(self, benchmark, is_real):
        """Benchmark `random_unitary` for both real and complex-valued matrices.

        Fixed Parameters:
            - `dim`: Set to a constant dimension of 64 for consistent comparison.
            - `seed`: Set to `None` so a new random seed is used for each run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            is_real (bool): If `True`, generate a real orthogonal matrix; otherwise, a complex unitary.
        """
        # Select a constant dimension.
        dim = 64

        result = benchmark(random_unitary, dim=dim, is_real=is_real, seed=None)

        assert result.shape == (dim, dim)

class TestRandomPsdOperatorBenchmarks:
    """Benchmarks for the `toqito.rand.random_psd_operator` function."""

    @pytest.mark.parametrize(
        "dim, is_real",
        [
            # Test across dimensions for both real and complex matrices.
            *itertools.product([2, 4, 8, 16, 32, 64, 128, 256], [True, False]),
        ],
        ids=lambda x: str(x)
    )
    def test_bench__random_psd_operator___vary___dim_is_real(self, benchmark, dim, is_real):
        """Benchmark `random_psd_operator` across various dimensions and for real/complex outputs.

        Fixed Parameters:
            - `seed`: Set to `None` so that a random seed is used for each benchmark run.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the positive semidefinite operator.
            is_real (bool): If `True`, generates a real-valued matrix; otherwise, complex.
        """
        result = benchmark(random_psd_operator, dim=dim, is_real=is_real, seed=None)

        assert result.shape == (dim, dim)

class TestTraceDistanceBenchmarks:
    """Benchmarks for the `toqito.state_metrics.trace_distance` function."""

    @pytest.mark.parametrize(
        "dim, matrix_type",
        [
            *itertools.product([4, 16, 64, 128, 256], ["identical", "random"]),
        ],
        ids = lambda x: str(x),
    )
    def test_bench__trace_distance__vary__rho_sigma(self, benchmark, dim, matrix_type):
        """Benchmark `trace_distance` for various matrix sizes and types.

        Fixed Parameters:
            - There are no fixed parameters for this benchmark.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension (n) for the n x n density matrices `rho` and `sigma`.
            matrix_type (str): Specifies if `rho` and `sigma` are "identical" or "random".
        """

        if matrix_type == "identical":
            mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat = mat @ mat.conj().T
            rho = np.divide(mat, np.trace(mat))
            result = benchmark(
                trace_distance,
                rho=rho, # A random nxn density matrix.
                sigma=rho # The same matrix as rho
            )
        elif matrix_type == "random":
            mat1 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat1 = mat1 @ mat1.conj().T
            rho = np.divide(mat1, np.trace(mat1))

            mat2 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat2 = mat2 @ mat2.conj().T
            sigma = np.divide(mat2, np.trace(mat2))

            result = benchmark(
                trace_distance,
                rho=rho,    # A random n x n density matrix.
                sigma=sigma # A different random n x n density matrix.
            )
        
        assert result is not None

class TestTraceNormBenchmarks:
    """Benchmarks for the `toqito.matrix_props` function."""
    @pytest.mark.parametrize(
        "dim, is_square",
        [
            (4, "square"),
            (16, "square"),
            (64, "square"),
            (128, "square"),
            (25, "not_square"),
            (100, "not_square"),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__trace_norm__vary__rho(self, benchmark, dim, is_square):
        """Benchmark `trace_norm` with varying matrix dimensions and square/non-square shapes.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension used to construct the input matrix `rho`.
            is_square (str): Indicates whether the generated `rho` should be "square" or "not_square".
        """

        rho = None
        if is_square == "not_square":
            # For "not_square", create a rectangular matrix (dim x 2*dim).
            rho = np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim)
            result = benchmark(
                trace_norm,
                rho=rho
            )

        elif is_square == "square":
            # For "square", create a square matrix (dim x dim).
            rho = np.random.rand(dim, 2*dim) + 1j*np.random.rand(dim, 2*dim)
            result = benchmark(
                trace_norm,
                rho=rho
            )
        assert result is not None

class TestLogNegativityBenchmarks:
    """Benchmarks for the `toqito.state_props.log_negativity` function."""

    @pytest.mark.parametrize(
        "rho_dim, dim_arg",
        [   
            (8, [2, 4]), 
            (8, [4, 2]), 
            (16, None), 
            (16, [4, 4]), 
            (16, [2, 8]), 
            (16, [8, 2]), 
            (64, None),
            (64, [8, 8]), 
            (64, [2, 32]), 
            (64, [32, 2]), 
            (128, 2), 
            (128, [2, 64]), 
            (128, [64, 2])
        ],
        ids = lambda x: str(x),
    )
    def test_bench__log_negativity__vary__rho_dim(self, benchmark, rho_dim, dim_arg):
        """Benchmark `log_negativity` with varying density matrix dimensions and subsystem dimensions.

        Fixed Parameters:
            - No parameters are fixed for this benchmark; all are varied.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            rho_dim (int): The dimension (n) of the n x n density matrix `rho`.
            dim_arg (list[int] | int | None): The `dim` argument passed to `log_negativity`, representing the dimensions of the subsystems.
        """
        mat1 = np.random.rand(rho_dim, rho_dim) + 1j * np.random.rand(rho_dim, rho_dim)
        mat1 = mat1 @ mat1.conj().T
        rho = np.divide(mat1, np.trace(mat1))

        result = benchmark(
            log_negativity,
            rho=rho,
            dim=dim_arg
        )
        assert isinstance(result, float)

class TestVonNeumannEntropyBenchmarks:
    """Benchmarks for the `toqito.state_props.von_neuman_entropy` function."""
    @pytest.mark.parametrize(
        "dim",
        [4, 16, 32, 64, 128, 256],
        ids = lambda x: str(x)
    )
    def test_bench__von_neumann_entropy__vary__rho(self, benchmark, dim):
        """Benchmark `von_neumann_entropy` by varying the dimension of the density matrix.

        Fixed Parameters:
            - None: All relevant parameters are varied through `pytest.mark.parametrize`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the square density matrix `rho`.
        """
        mat1 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        mat1 = mat1 @ mat1.conj().T
        rho = np.divide(mat1, np.trace(mat1))

        result = benchmark(von_neumann_entropy, rho=rho)

class TestNaturalRepresentationBenchmarks:
    """Benchmarks for the `toqito.channel_ops.natural_representation` function."""

    @pytest.mark.parametrize(
        "dim, num_ops",
        [
            (4, 2),
            (4, 32),
            (8, 2),
            (8, 64),
            (16, 2),
            (16, 128),
            (32, 2),
            (32, 16),
        ],
        ids=lambda x: str(x)
    )
    def test_bench__natural_representation__vary__kraus_ops(self, benchmark, dim, num_ops):
        """Benchmark `natural_representation` with varying Kraus operator dimensions and number of operators.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of each individual Kraus operator.
            num_ops (int): The number of Kraus operators in the list.
        """
        
        # Generate a list of random complex Kraus operators with specified dimensions.
        kraus_ops = [np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim) for _ in range(num_ops)]

        result = benchmark(
            natural_representation,
            kraus_ops=kraus_ops
        )
        assert result.shape == (dim**2, dim**2)

class TestAmplitudeDampingBenchmarks:
    """Benchmarks for the `toqito.channels.amplitude_damping` function."""

    @pytest.mark.parametrize(
        "input_mat, gamma, prob",
        [
            (False, 0.0, 0.0),
            (False, 0.1, 0.5),
            (False, 0.5, 0.5),
            (False, 0.7, 0.2),
            (False, 0.1, 1.0),
            (False, 0.7, 1.0),
            (False, 1.0, 1.0),
            (True, 0.0, 0.0),
            (True, 0.1, 0.5),
            (True, 0.5, 0.5),
            (True, 0.7, 0.),
            (True, 1.0, 1.0),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__amplitude_damping__vary__input_mat_gamma_prob(self, benchmark, input_mat, gamma, prob):
        """Benchmark `amplitude_damping` with varying input matrix presence, damping rate, and probability.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            input_mat (bool): A boolean indicating whether a 2x2 input matrix should be generated (`True`)
                              or if the function should return Kraus operators (`False`).
            gamma (float): The damping rate for the amplitude damping channel.
            prob (float): The probability of energy loss for the generalized amplitude damping channel.
        """

        if input_mat:
            mat = np.random.rand(2,2) + 1j * np.random.rand(2,2)
            input_mat = mat @ mat.conj().T
            input_mat = input_mat/np.trace(input_mat)
            result = benchmark(
                amplitude_damping,
                input_mat=input_mat,
                gamma=gamma,
                prob=prob
            )
            assert np.isclose(np.trace(result), 1.0)
        else:
            result = benchmark(
                amplitude_damping,
                input_mat=None,
                gamma=gamma,
                prob=prob
            )
            assert len(result) == 4
    
class TestBitflipBenchmarks:
    """Benchmarks for the `toqito.channels.bitflip` function."""

    @pytest.mark.parametrize(
        "input_mat, prob",
        [
            (False,  0.0),
            (False, 0.2),
            (False, 0.8),
            (False, 1.0),
            (True, 0.0),
            (True, 0.2),
            (True, 0.8),
            (True, 1.0),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__bitflip__vary__input_mat_prob(self, benchmark, input_mat, prob):
        """Benchmark `bitflip` with varying input matrix presence and bitflip probability.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            input_mat (bool): A boolean flag: if `True`, a 2x2 density matrix is used as input;
                              if `False`, `None` is passed as input to get Kraus operators.
            prob (float): The probability of the bitflip occurring.
        """
        if input_mat:
            # Generate a random 2x2 density matrix as input.
            mat = np.random.rand(2,2) + 1j * np.random.rand(2,2)
            input_mat = mat @ mat.conj().T
            input_mat = input_mat/np.trace(input_mat)

            result = benchmark(bitflip, input_mat = input_mat, prob = prob)
            assert np.isclose(np.trace(result), 1.0)
        else:
            # When no input matrix is provided, the function should return 2 Kraus operators.
            result = benchmark(bitflip, input_mat = None, prob = prob)
            assert len(result) == 4

class TestDephasingBenchmarks:
    """Benchmarks for `toqito.channels.dephasing` function."""

    @pytest.mark.parametrize(
        "dim",
        [2, 4, 8, 16, 32, 64],
        ids = lambda x : str(x)
    )
    def test_bench__dephasing__vary__dim(self, benchmark, dim):
        """Benchmark `dephasing` with varying dimensionality.

        Fixed Parameters:
            - `param_p`: Set to `0` to produce the completely dephasing channel.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimensionality of the channel.
        """

        choi_mat = benchmark(dephasing, dim = dim, param_p = 0)
        assert choi_mat.shape == (dim*dim, dim*dim)
    
    @pytest.mark.parametrize(
        "param_p",
        [0.0, 0.1, 0.5, 0.9, 1.0],
        ids = lambda x: str(x)
    )
    def test_bench__dephasing__vary__param_p(self, benchmark, param_p):
        """Benchmark `dephasing` with varying probability parameter `param_p`.

        Fixed Parameters:
            - `dim`: A constant dimension of 32 is used for the channel.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            param_p (float): The probability parameter for the dephasing channel.
        """
        dim = 16 # Select a constant dimension.
        choi_mat = benchmark(dephasing, dim=dim, param_p=param_p)
        assert choi_mat.shape == (dim * dim, dim * dim)
    
    @pytest.mark.parametrize(
        "dim, param_p",
        [
            (4, 0.2),
            (4, 0.7),
            (16, 0.2),
            (16, 0.7),
            (64, 0.2),
            (64, 0.7),
        ],
        ids=lambda x: str(x)
    )
    def test_bench__dephasing__dim_param_p(self, benchmark, dim, param_p):
        """Benchmark `dephasing` for a combination of dimension and probability parameter.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimensionality of the channel.
            param_p (float): The probability parameter for the dephasing channel.
        """
        choi_mat = benchmark(dephasing, dim=dim, param_p=param_p)
        assert choi_mat.shape == (dim * dim, dim * dim)

class TestBasisBenchmarks:
    """Benchmarks for the `toqito.states.basis` function."""

    @pytest.mark.parametrize(
        "dim",
        [4, 16, 64, 256],
        ids = lambda x: str(x),
    )
    def test_bench__basis__vary__dim(self, benchmark, dim):
        """Benchmark `basis` with varying dim.

        Fixed Parameters:
            - `pos`: Set to 2.
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the output basis.
        """
        
        result = benchmark(basis, dim=dim, pos = 2)

        assert result.shape[0] == dim
    
class TestIsPositiveSemidefiniteBenchmarks:

    """Benchmarks for the `toqito.matrix_props.is_positive_semidefinite` function."""
    @pytest.mark.parametrize(
        "matrix_size, is_psd_real",
        [
            *itertools.product([4, 16, 64, 256], [True, False]),
        ],
        ids=lambda x: str(x)
    )
    def test_bench__is_positive_semidefinite__vary__mat(self, benchmark, matrix_size, is_psd_real):
        """Benchmark `is_positive_semidefinite` with varying matrix sizes and PSD property.

        Fixed Parameters:
            - None

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (N) of the N x N input matrix.
            is_psd_real (bool): Whether the generated matrix should be positive semidefinite.
        """
        if is_psd_real:
            rand_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
            mat = rand_mat @ rand_mat.conj().T
            mat = rand_mat @ rand_mat.conj().T
        else:
            rand_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
            hermitian_mat = (rand_mat + rand_mat.conj().T) / 2
            mat = hermitian_mat - np.eye(matrix_size) * (np.max(np.abs(hermitian_mat)) + 1)
        
        result = benchmark(is_positive_semidefinite, mat=mat)
        assert result == is_psd_real

class TestSwapBenchmarks:


    """Benchmarks for the `toqito.perms.swap` function."""

    @pytest.mark.parametrize(
        "matrix_size",
        [4, 16, 64, 256],
        ids = lambda x: str(x),
    )
    def test_bench__swap__vary__rho(self, benchmark, matrix_size):
        """Benchmark `swap` with varying input matrix sizes (square matrices).

        Fixed Parameters:
            - `sys`: Set to `None` to use default behavior, which swaps the first two subsystems.
            - `dim`: Set to `None` to use default behavior, which infers subsystem dimensions as square systems of size `sqrt(len(rho))`.
            - `row_only`: False

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (n) of the n x n input matrix.
        """
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        result = benchmark(swap, rho=input_mat, sys=None, dim=None, row_only=False)

        assert result.shape == (matrix_size, matrix_size) 
    

    @pytest.mark.parametrize(
        "num_subsystems, sys",
        [
            (2, [1, 2]),
            (3, [1, 2]),
            (3, [1, 3]),
            (3, [2, 3]),
            (4, [1, 2]),
            (4, [2, 3]),
            (4, [1, 4]),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__swap__vary__sys_dim(self, benchmark, num_subsystems, sys):
        """Benchmark `swap` by varying the systems to swap (`sys`) and implicitly the dimensions (`dim`).

        Fixed Parameters:
            - `rho`: Generated as a random complex matrix with size determined by `num_subsystems` and fixed subsystem dimension of 2.
            - `row_only`: False

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            num_subsystems (int): The total number of subsystems.
            sys (list[int]): The two subsystems to be swapped.
        """
        # Assume each subsystem has dimension 2.
        dim = [2]* num_subsystems
        matrix_size = int(np.prod(dim))
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        
        result = benchmark(swap, rho=input_mat, sys=sys, dim=dim, row_only=False)

        assert result.shape == (matrix_size, matrix_size)
    

    @pytest.mark.parametrize(
        "dim",
        [
            4,
            16,
            64,
            [2, 2],
            [4, 4],
            [8, 8],
            [4, 4, 4, 4],
        ],
        ids = lambda x: str(x)
    )
    def test_bench__swap__vary__dim(self, benchmark, dim):
        """Benchmark `swap` by varying subsystem dimensions (`dim`).

        Fixed Parameters:
            - `rho`: Generated as a random complex matrix with size determined by `dim`.
            - `sys`: Set to `None` to use default behavior, which swaps the first two subsystems.
            - `row_only`: False

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int | list[int]): The dimension parameter for the `dim` argument.
        """
        if isinstance(dim, int):
            matrix_size = dim**2
        else:
            matrix_size = int(np.prod(dim))
        
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        result = benchmark(swap, rho=input_mat, sys=None, dim=dim, row_only=False)

        assert result.shape == (matrix_size, matrix_size)


    @pytest.mark.parametrize(
        "row_only",
        [False, True],
        ids = lambda x: str(x)
    )
    def test_bench__swap__vary__rho_only(self, benchmark, row_only):
        """Benchmark `swap` with varying the `row_only` parameter.

        Fixed Parameters:
            - `rho`: A fixed-size square matrix (64x64).
            - `sys`: Set to `None` to use default behavior, which swaps the first two subsystems.
            - `dim`: Set to `None` to use default behavior, which infers subsystem dimensions.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            row_only (bool): Boolean value for the `row_only` parameter.
        """
        matrix_size = 64
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        result = benchmark(swap, rho=input_mat, sys=None, dim=None, row_only=row_only)

        assert result.shape == (matrix_size, matrix_size)

class TestSwapOperatorBenchmarks:
    """Benchmarks for the `toqito.perms.swap_operator` function."""

    @pytest.mark.parametrize(
        "dim, is_sparse",
        [
            (8, False),
            (8, True),
            ([4, 4], False),
            ([4, 4], True),
            ([2, 2, 2], False),
            ([2, 2, 2], True),
        ],
        ids=lambda x: str(x),
    )
    def test_bench__swap_operator__vary__dim_is_sparse(self, benchmark, dim, is_sparse):
        """Benchmark `swap_operator` with varying dimensions and `is_sparse` flag.

        Fixed Parameters:
            - None.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int | list[int]): The dimension(s) of the subsystems.
            is_sparse (bool): Boolean indicating whether the operator should be sparse.
        """
        result = benchmark(swap_operator, dim=dim, is_sparse=is_sparse)

        if isinstance(dim, int):
            expected_size = dim**2
        else:
            expected_size = int(np.prod(dim))

        if is_sparse:
            # The `swap_operator` function, due to its internal call to `swap` and `permute_systems`,
            # currently always returns a dense numpy array, even if `is_sparse` is True.
            
            #assert isinstance(result, (sp.sparse.csc.csc_matrix, sp.sparse.bsr.bsr_matrix))
            assert result.shape == (expected_size, expected_size)
        else:
            assert isinstance(result, np.ndarray)
            assert result.shape == (expected_size, expected_size)

class TestToDensityMatrixBenchmarks:
    """Benchmarks for the `toqito.matrix_ops.to_density_matrix` function."""

    @pytest.mark.parametrize(
        "input_type, size",
        [
            *itertools.product(["vector", "matrix"], [4, 16, 64, 256]),
        ],
        ids = lambda x:str(x)
    )
    def test_bench__to_density_matrix__vary__input_array(self, benchmark, input_type, size):
        """Benchmark `to_density_matrix` with varying input array types and sizes.

        Fixed Parameters:
            - None. 

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            input_type (str): Specifies the type of input array to generate ("vector" or "matrix").
            size (int): The dimension for the input vector or matrix (e.g., N for an N x 1 vector or N x N matrix).
        """
        input_array = None

        if input_type == "vector":
            input_array = np.random.rand(size, 1) + 1j * np.random.rand(size, 1)
        elif input_type == "matrix":
            input_array = np.random.rand(size, size) + 1j * np.random.rand(size, size)

        result = benchmark(to_density_matrix, input_array)

        assert result.shape == (size, size)

class TestPermuteSystemsBenchmarks:
    """Benchmarks for the `toqito.perms.permute_systems` function."""

    @pytest.mark.parametrize(
        "dim, perm",
        [
            # 2 subsystems
            ([2, 2], [1, 0]),  # 4x4 matrix
            ([4, 4], [1, 0]),  # 16x16 matrix
            ([8, 8], [1, 0]),  # 64x64 matrix
            # 3 subsystems
            ([2, 2, 2], [1, 2, 0]),  # 8x8 matrix
            ([4, 4, 4], [2, 0, 1]),  # 64x64 matrix
            # 4 subsystems
            ([2, 2, 2, 2], [3, 2, 1, 0]),  # 16x16 matrix
            ([4, 4, 4, 4], [3, 0, 2, 1]),  # 256x256 matrix
        ],
        ids = lambda x:str(x)
    )
    def test_bench__permute_systems__vary__dim_perm(self, benchmark, dim, perm):

        matrix_size = int(np.prod(dim))
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size) 
        result = benchmark(permute_systems, input_mat=input_mat, perm=perm, dim=dim, row_only = False, inv_perm = False)

        assert result.shape == (matrix_size, matrix_size)
    

    @pytest.mark.parametrize(
        "row_only, inv_perm",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
        ids = lambda x:str(x)
    )
    def test_bench__permute_systems__vary__row_only_inv_perm(self, benchmark, row_only, inv_perm):

        dim = [8, 8]
        perm = [1, 0]
        matrix_size = int(np.prod(dim))
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size) 
        result = benchmark(
            permute_systems,
            input_mat=input_mat,
            perm=perm,
            dim=dim,
            row_only=row_only,
            inv_perm=inv_perm,
        )

        assert result.shape == (matrix_size, matrix_size)

    @pytest.mark.parametrize(
        "setup",
        [
            {"size": 4, "dim": [2, 2], "perm": [1, 0]},
            {"size": 8, "dim": [2, 2, 2], "perm": [1, 2, 0]},
            {"size": 16, "dim": [4, 4], "perm": [1, 0]},
            {"size": 64, "dim": [4, 4, 4], "perm": [2, 1, 0]},
            {"size": 256, "dim": [16, 16], "perm": [1, 0]},
        ],
        ids=lambda x:str(x)
    )
    def test_bench__permute_systems__vary__vector_input(self, benchmark, setup):

        size = setup["size"]
        input_mat = np.random.rand(size, 1) + np.random.rand(size, 1)

        result = benchmark(permute_systems, input_mat=input_mat, perm=setup["perm"], dim=setup["dim"])
        assert result.shape == (size,) 
    
class TestKraustoChoiBenchmarks:
    """Benchmarks for the `toqito.channel_ops.kraus_to_choi` function."""

    @pytest.mark.parametrize(
        "dim, num_ops, cp",
        [
            (2, 4, True),
            (2, 4, False),
            (2, 32, True),
            (2, 32, False),
            (4, 4, True),
            (4, 4, False),
            (4, 32, True),
            (4, 32, False),
            (16, 4, True),
            (16, 4, False),
            (16, 32, True),
            (16, 32, False),
        ],
    )
    def test_bench__kraus_to_choi__vary__kraus_ops(self, benchmark, dim, num_ops, cp):
        """Benchmark `kraus_to_choi` by varying the size, number, and structure of Kraus operators.

        Fixed Parameters:
            - `sys`: Set to the default value of `2`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the square Kraus operators.
            num_ops (int): The number of base Kraus operators to generate.
            cp (bool): If `True`, the channel is treated as completely positive (flat list of operators).
                       If `False`, it's not (list of `[A, A.conj().T]` pairs).
        """
        base_ops = [np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim) for _ in range(num_ops)]
        if cp:
            # For a completely positive (CP) map, the input is a flat list of Kraus operators.
            kraus_ops = base_ops
        else:
            # For a map that is not completely positive, the input is a list of pairs [A, B].
            kraus_ops = [[op, op.conj().T] for op in base_ops]

        result = benchmark(kraus_to_choi, kraus_ops=kraus_ops, sys=2)
        assert result.shape == (dim**2, dim**2)
    
    @pytest.mark.parametrize("sys", [1, 2])
    def test_bench__kraus_to_choi__param__sys(self, benchmark, sys):
        """Benchmark `kraus_to_choi` by varying the `sys` parameter.

        Fixed Parameters:
            - `kraus_ops`: Fixed set of 4 non-completely positive Kraus operators of size 16x16.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sys (int): The system on which the channel is applied (1 or 2).
        """
        dim = 16
        num_ops = 4
        base_ops = [np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim) for _ in range(num_ops)]
        kraus_ops = [[op, op.conj().T] for op in base_ops]

        result = benchmark(kraus_to_choi, kraus_ops=kraus_ops, sys=sys)
        assert result.shape == (dim**2, dim**2)

class TestPauliBenchmarks:
    """Benchmarks for the `toqito.matrices.pauli` function."""

    @pytest.mark.parametrize(
        "type, ind",
        [
            *itertools.product(["int"], [0,1,2,3]),
            *itertools.product(["str"], ['I', 'X', 'Y', 'Z']),
            *itertools.product(["list"], [2, 4, 8]),
        ]
    )
    def test_bench__pauli__vary__ind(self, benchmark, type, ind):
        """Benchmark `pauli` by varying the `ind` parameter type and value.

        Fixed Parameters:
            - `is_sparse`: Set to `False` to always generate dense numpy arrays.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            type (str): A helper string ("int", "str", "list") to control logic.
            ind (int | str | list[int]): The index for the Pauli operator. When `type` is "list",
                                         `ind` represents the number of qubits (length of the list).
        """
        

        if type == "int" or type == "str":
            # Benchmark generating a single 2x2 Pauli matrix.
            result = benchmark(pauli, ind=ind, is_sparse = False)
            assert result.shape == (2,2)
        if type == "list":
            # When type is "list", `ind` is the number of qubits for the tensor product.
            ind = np.random.randint(0,4, size = ind).tolist()
            result = benchmark(pauli, ind=ind, is_sparse = False)
            assert result.shape == (2**len(ind), 2**len(ind))
            
    @pytest.mark.parametrize(
        "is_sparse",
        [True, False],
        ids= lambda x: str(x)
    )
    def test_bench__pauli__param__is_sparse(self, benchmark, is_sparse):
        """Benchmark `pauli` by varying the `is_sparse` parameter.

        Fixed Parameters:
            - `ind`: A randomly generated list of 8 indices, creating a 256x256 operator.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            is_sparse (bool): The flag to determine if the output should be sparse or dense.
        """
        # A list of length 8 creates a 2^8 x 2^8 = 256x256 matrix.
        size = 8

        ind = np.random.randint(0, 4, size = size).tolist()
        result = benchmark(pauli, ind = ind, is_sparse=is_sparse)
        if not is_sparse:
            assert result.shape == (2**size, 2**size)
        else:
            assert result.shape == (2,2)

class TestPermutationOperatorBenchmarks:
    """Benchmarks for the `toqito.perms.permutation_operator` function."""

    @pytest.mark.parametrize(
        "dim , perm",
        [
            (2, [1, 0]),  # small dim
            (2, [1, 2, 0]),
            (2, [1, 2, 0, 3]),
            (2, [1, 2, 0, 3, 4]),
            (2, [1, 2, 5, 0, 3, 4]),
            (16, [1, 0]),  # small dim
            (16, [1, 2, 0]),
            (8, [1, 2, 0, 3]),
        ],
        ids=lambda x: str(x),
    )
    def test_bench_permutation_operator_vary_dim_perm(self, benchmark, dim, perm):
        """Benchmark `permutation_operator` with varying `dim` and `perm` parameters.

        Fixed Parameters:
            - `inv_perm`: Set to `False` to test the non-inverse permutation.
            - `is_sparse`: Set to `False` to return a dense NumPy array.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the subsystems.
            perm (list[int]): A permutation vector.
        """
        result = benchmark(permutation_operator, dim=dim, perm=perm, inv_perm=False, is_sparse=False)

        assert result.shape == (dim ** (max(perm) + 1), dim ** (max(perm) + 1))
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        "dim, perm, is_sparse",
        [
            (8, [0, 1, 3, 2], True),
            (8, [0, 1, 3, 2], False),
        ],
        ids=lambda x: str(x),
    )
    def test_bench_permutation_operator_param_is_sparse(self, benchmark, dim, perm, is_sparse):
        """Benchmark `permutation_operator` with varying `is_sparse` parameter.

        Fixed Parameters:
            - `dim`: Set to `8` .
            - `perm`: Set to `[0, 1, 3, 2]` as a representative permutation.
            - `inv_perm`: Set to `False` to test the non-inverse permutation.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the subsystems.
            perm (list[int]): A permutation vector.
            is_sparse (bool): Boolean indicating if the return is sparse or not.
        """
        result = benchmark(permutation_operator, dim=dim, perm=perm, inv_perm=False, is_sparse=is_sparse)
        if is_sparse:
            # the output returned is not sparse here, stick to np.ndarray
            #assert sp.sparse.issparse(result)
            assert isinstance(result, np.ndarray)
        else:
            assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        "dim, perm, inv_perm",
        [
            (8, [0, 1, 3, 2], True),
            (8, [0, 1, 3, 2], False),
        ],
        ids=lambda x: str(x),
    )
    def test_bench_permutation_operator_param_inv_perm(self, benchmark, dim, perm, inv_perm):
        """Benchmark `permutation_operator` with varying `inv_perm` parameter.

        Fixed Parameters:
            - `dim`: Set to `8`.
            - `perm`: Set to `[0, 1, 3, 2]` as a representative permutation.
            - `is_sparse`: Set to `False` to return a dense NumPy array.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the subsystems.
            perm (list[int]): A permutation vector.
            inv_perm (bool): boolean representing if `perm` is inverse or not.
        """
        result = benchmark(permutation_operator, dim=dim, perm=perm, inv_perm=inv_perm, is_sparse=False)

        assert result.shape == (dim ** (max(perm) + 1), dim ** (max(perm) + 1))

class TestApplyChannelBenchmarks:
    """Benchmarks for the `toqito.channel_ops.apply_channel` function."""

    @pytest.mark.parametrize(
        "phi_op_type, dim",
        [
            ("kraus", 4),
            ("kraus", 16),
            ("kraus", 64),
            ("kraus", 256),
            ("choi", 2),
            ("choi", 4),
            ("choi", 16),
            ("choi", 64),
            ("choi", (4, 16)),
            ("choi", (32, 64)),
        ],
        ids = lambda x:str(x)
    )
    def test_bench__apply_channel__vary__phi_op(self, benchmark, phi_op_type, dim):
        """Benchmark `apply_channel` with varying `phi_op` types (Kraus or Choi) and dimensions.

        Fixed Parameters:
            - None: All parameters are varied.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            phi_op_type (str): The type of `phi_op` to use ('kraus' or 'choi').
            dim (int | tuple[int, int]): The dimension of the input matrix and/or the system for `phi_op`.
        """

        if phi_op_type == "kraus":
            # For Kraus operators, input_mat is always square.
            input_mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            # choice for Kraus operators is 4.
            phi_op = [np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim) for _ in range(4)]
            
            expected_shape = (dim, dim)
        elif phi_op_type == "choi":
            if isinstance(dim, int):
                input_mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
                # Use swap_operator to generate a Choi matrix for square systems.
                phi_op = swap_operator(dim)
                expected_shape = (dim, dim)
            else:  # This handles the tuple case, e.g., (3, 2)
                input_mat = np.random.randn(*dim) + 1j * np.random.randn(*dim)
                # Use swap_operator with list(dim) for non-square systems.
                phi_op = swap_operator(list(dim))
                expected_shape = tuple(reversed(dim))

        result = benchmark(apply_channel, mat=input_mat, phi_op=phi_op)

        assert result.shape == expected_shape

class TestPartialTransposeBenchmarks:
    """Benchmarks for the `toqito.channels.partial_transpose` function."""

    @pytest.mark.parametrize(
        "dim, is_cvxpy_var",
        [*itertools.product([4, 16, 64, 256], [False, True])],
        ids=lambda x: str(x),
    )
    def test_bench__partial_transpose__vary__rho(self, benchmark, dim, is_cvxpy_var):
        """Benchmark `partial_transpose` with varying input matrix dimensions and CVXPY variable types.

        Fixed Parameters:
            - `sys`: Set to `None` to use the default behavior of transposing the second subsystem.
            - `dim`: Set to `None` to infer subsystem dimensions automatically (assuming square subsystems).

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the square input matrix `rho`.
            is_cvxpy_var (bool): A boolean indicating whether `rho` should be a CVXPY variable or a NumPy array.
        """
        rho_np = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)

        if is_cvxpy_var:
            rho = Variable((dim, dim), complex=True)
        else:
            rho = rho_np

        result = benchmark(partial_transpose, rho=rho, sys=None, dim=None)

        assert result is not None

    @pytest.mark.parametrize(
        "sys",
        [
            0,
            1,
            2,
            [0, 1],
            [0, 1, 2],
        ],
        ids=lambda x: str(x),
    )
    def test_bench__partial_transpose__vary__sys(self, benchmark, sys):
        """Benchmark `partial_transpose` with varying `sys` (subsystem(s) to transpose).

        Fixed Parameters:
            - `rho`: A fixed `8x8` NumPy array.
            - `dim`: Set to `[2, 2, 2]` to represent a 3-subsystem system, where each subsystem has dimension 2.
            - `is_cvxpy_var`: Set to `False` to use a NumPy array for `rho`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sys (int | list[int]): The index or list of indices of the subsystems to transpose.
        """
        dim = 8
        rho = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        subsystem_dims = [2, 2, 2]  # Represents an 8x8 matrix composed of 3 subsystems of dim 2 each.

        result = benchmark(partial_transpose, rho=rho, sys=sys, dim=subsystem_dims)

        assert result is not None

    @pytest.mark.parametrize(
        "sub_dim",
        [
            [[8, 8], [8, 8]],
            [[4, 16], [16, 4]],
            [[4, 4, 4], [4, 4, 4]],
            [[2, 4, 8], [8, 2, 4]],
            [[2, 2, 4, 4], [2, 2, 4, 4]],
            [[2, 8, 2, 2], [2, 2, 8, 2]],
            [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ],
        ids=lambda x: str(x),
    )
    def test_bench__partial_transpose__vary__dim(self, benchmark, sub_dim):
        """Benchmark `partial_transpose` with varying `dim` (subsystem dimensions).

        Fixed Parameters:
            - `rho`: A fixed `64x64` NumPy array. Chosen to be large enough to accommodate various `sub_dim` configurations.
            - `sys`: Set to `None` to use the default behavior of transposing the second subsystem, based on the `dim` provided.
            - `is_cvxpy_var`: Set to `False` to use a NumPy array for `rho`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sub_dim (list[list[int]]): A list specifying the dimensions of the subsystems for rows and columns.
        """
        # The total dimension of `rho` must be consistent with the product of `sub_dim` elements.
        # For a 64x64 matrix, possible `sub_dim` products include 8*8, 4*16, 2*4*8, etc.
        dim = 64
        rho = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)

        result = benchmark(partial_transpose, rho=rho, sys=None, dim=sub_dim)

        assert result is not None

class TestVecBenchmarks:
    """Benchmarks for the `toqito.perms.vec` function."""

    @pytest.mark.parametrize(
        "matrix_size",
        [4, 16, 64, 256],
        ids = lambda x: str(x),
    )
    def test_bench__vec__vary__mat(self, benchmark, matrix_size):
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        result = benchmark(vec, mat = input_mat)

        assert result.shape[1] == 1