import pytest
import numpy as np
import itertools

from toqito.channels import partial_trace


from toqito.rand import random_density_matrix
from toqito.rand import random_unitary
from toqito.rand import random_psd_operator

from toqito.state_metrics import trace_distance

from toqito.matrix_props import trace_norm
from toqito.matrix_props import is_positive_semidefinite

from toqito.state_props import log_negativity
from toqito.state_props import von_neumann_entropy

from toqito.channel_ops import natural_representation

from toqito.channels import amplitude_damping
from toqito.channels import bitflip
from toqito.channels import dephasing

from toqito.states import basis

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