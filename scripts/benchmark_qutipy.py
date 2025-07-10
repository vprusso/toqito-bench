import pytest
import numpy as np
import itertools

from qutipy.general_functions import partial_trace
from qutipy.general_functions import trace_norm
from qutipy.states import random_density_matrix
from qutipy.states import log_negativity
from qutipy.gates import RandomUnitary
from qutipy.general_functions import random_PSD_operator
from qutipy.distance import norm_trace_dist
from qutipy.entropies import entropy
from qutipy.channels import natural_representation

class TestPartialTraceBenchmarks:
    """Benchmarks for the `qutipy.general_functions.partial_trace` function"""
    
    @pytest.mark.parametrize("matrix_size", [4, 16, 64, 256], ids = lambda x: str(x))
    def test_bench__partial_trace__vary__input_mat(self, benchmark, matrix_size):
        """Benchmark `partial_trace` with varying input matrix sizes.

        Fixed Parameters:
            - `sys`: Set to [2] to take trace over the second subsystem.
            - `dim`: Set as a list of square roots of the `matrix_size`
        
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            matrix_size (int): The dimension (n) of the n x n input matrix.
        """

        dim = [int(np.sqrt(matrix_size)), int(np.sqrt(matrix_size))]

        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        sys = [2]

        result = benchmark(partial_trace, input_mat, sys = sys, dim = dim)

        assert result.shape[0] <= matrix_size
    
    @pytest.mark.parametrize(
        "sys_param",
        [
            [0],
            [1],
            [0,1],
            [0,2],
        ],
        ids=lambda x:str(x),
    )
    def test_bench__partial_trace__vary__sys(self, benchmark, sys_param):
        """Benchmark `partial_trace` by tracing out different subsystems.

        Fixed Parameters:
            - `input_mat`: Generated with a constant matrix size of `16 x 16`.
            - `dim`: Dynmaically set to be compatible with `sys` value.
        
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sys (list[int]) : system to take trace of.
        """
        # Set the dim by calibrating sys to 1 based indexing
        sys = [x + 1 for x in sys_param]
        pair = {(1,): [4, 4], (2,): [4, 4], (1, 3): [2, 2, 2, 2], (1, 2): [4, 4]}
        dim = pair[tuple(sys)]

        matrix_size = int(np.prod(dim))
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        result = benchmark(partial_trace, input_mat, sys=sys, dim=dim)   
        assert result is not None
    
    @pytest.mark.parametrize(
        "dim",
        [
            [16, 1],
            [2, 2],
            [2, 2, 2, 2],
            [3, 3],
            [4, 4]
        ],
        ids = ["None", "[2, 2]", "[2, 2, 2, 2]", "[3, 3]", "[4, 4]"]
    )
    def test_bench__partial_trace__vary__dim(self, benchmark, dim):
        """Benchmark `partial_trace` by varying subsystem dimensions (`dim`).

        Fixed Parameters:
            - `input_mat`: Random matrix generated with size set as product of the dimension in `dim`
            - `sys`: Set to [2] to take trace over the second subsystem.
        
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (list[int] | None): A list specifying the dimension of each subsystem.
        """
        
        matrix_size = int(np.prod(dim))

        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        sys = [2]

        result = benchmark(partial_trace, input_mat, sys=sys, dim=dim)
        assert result is not None


class TestRandomDensityMatrixBenchmarks:
    """Benchmarks for the `qutipy.states.partial_trace function`"""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024], ids = lambda x: str(x))
    def test_bench__random_density_matrix__vary__dim(self, benchmark, dim):
        """Benchmark `random_density_matrix` with varying output sizes.

        Fixed Parameters:
            - `r`(rank): Not supplied, so the function internally sets `r = dim` to generate full-rank density matrices.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the output density matrix.
        """
        result = benchmark(random_density_matrix, dim = dim)
        
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
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the output density matrix.
            k_param (int): The rank of the density matrix.

        """
        result = benchmark(random_density_matrix, dim, k_param)
        assert result.shape == (dim, dim)


class TestRandomUnitaryBenchmarks:
    """Benchmarks for the `qutipy.gates.RandomUnitary` function"""

    @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024], ids=lambda x: str(x))
    def test_bench__random_unitary__vary__dim(self, benchmark, dim):
        """Benchmark `RandomUnitary` while varying the unitary matrix dimension.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension (n) of the generated n x n unitary matrix.
        
        """

        result = benchmark(RandomUnitary, dim = dim)

        assert result.shape == (dim, dim)

class TestRandomPSDOperatorBenchmarks:
    """Benchmarks for the `qutipy.general_functions.random_PSD_operator` function"""

    @pytest.mark.parametrize(
        "dim",
        [2, 4, 8, 16, 32, 64, 128, 256],
        ids = lambda x: str(x) + "-False",
    )
    def test_bench__random_psd_operator__vary__dim_is_real(self, benchmark, dim):
        """Benchmark `random_PSD_operator` while varying the generated operator's dimension

        Fixed Parameters:
            - `normal` (bool): Set to `False` so each matrix element is drawn from the uniform distribution.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension (n) of the generated n x n positive semi-definite operator. 
        """

        result = benchmark(random_PSD_operator, d=dim, normal = False)
        
        assert result.shape == (dim, dim)


class TestTraceDistanceBenchmarks:
    """Benchmarks for the `qutipy.distance.norm_trace_dist` function."""

    @pytest.mark.parametrize(
        "dim, matrix_type",
        [
            *itertools.product([4, 16, 64, 128, 256], ["identical", "random"]),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__trace_distance__vary__rho_sigma(self, benchmark, dim, matrix_type):
        """Benchmark `norm_trace_dist` by varying dimensions and matrix types of rho and sigma.

        Fixed Parameters:
            - `sdp`: Set to `False` to use the trace norm calculation instead of the SDP solver.
            - `dual`: Set to `False` (only relevant when `sdp=True`).
            - `display`: Set to `False` to suppress solver output.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension (n) of the n x n quantum states `rho` and `sigma`.
            matrix_type (str): Specifies whether `rho` and `sigma` are "identical" or "random".
        """
        if matrix_type == "identical":
            mat = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat = mat @ mat.conj().T
            rho = np.divide(mat, np.trace(mat))

            result = benchmark(
                norm_trace_dist,
                rho=rho,
                sigma=rho,
                sdp=False, # Fixed to use the trace_norm method.
                dual=False, # Not applicable when sdp=False, but kept for completeness.
                display=False # Fixed to suppress solver display.
            )
        elif matrix_type == "random":
            mat1 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat1 = mat1 @ mat1.conj().T
            rho = np.divide(mat1, np.trace(mat1))

            mat2 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            mat2 = mat2 @ mat2.conj().T
            sigma = np.divide(mat2, np.trace(mat2))
            result = benchmark(
                norm_trace_dist,
                rho=rho,
                sigma=sigma,
                sdp=False, # Fixed to use the trace_norm method.
                dual=False, # Not applicable when sdp=False, but kept for completeness.
                display=False # Fixed to suppress solver display.
            )

        assert result is not None

class TestTraceNormBenchmarks:
    """Benchmarks for the `qutipy.general_functions.trace_norm` function."""
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
        ids = lambda x : str(x)
    )
    def test_bench__trace_norm__vary__rho(self, benchmark, dim, is_square):
        """Benchmark `trace_norm` with varying matrix dimensions and square/non-square shapes.

        Fixed Parameters:
            - `sdp`: Set to `False` to utilize NumPy's `norm` function for trace norm calculation.
            - `dual`: Set to `False` (only relevant when `sdp=True`).

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The base dimension used for creating the input matrix `X`.
            is_square (str): Determines whether the generated matrix `X` will be "square" or "not_square".
        """
        X = None
        if is_square == "not_square":
            # For "not_square", create a rectangular matrix (dim x 2*dim).
            X = np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim)
            result = benchmark(
                trace_norm,
                X=X,
                sdp=False, # Fixed to use the standard NumPy trace norm.
                dual=False # This parameter is ignored when sdp=False.
            )

        elif is_square == "square":
            # For "square", create a square matrix (dim x dim).
            X = np.random.rand(dim, 2*dim) + 1j*np.random.rand(dim, 2*dim)
            result = benchmark(
                trace_norm,
                X=X,
                sdp=False, # Fixed to use the standard NumPy trace norm.
                dual=False # This parameter is ignored when sdp=False.
            )
        
        assert result is not None

class TestLogNegativityBenchmarks:
    """Benchmarks for the `qutipy.states.log_negativity` function."""

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
    def test_bench_log_negativity_vary_rho_dim(self, benchmark, rho_dim, dim_arg):
        """Benchmark `log_negativity` by varying the total dimension of the state and the subsystem dimensions.

        Fixed Parameters:
            - None

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            rho_dim (int): The total dimension of the bipartite state `rhoAB`.
            dim_arg (list or int or None): Specifies how `dimA` and `dimB` are determined.
                                           If `None`, `dimA` and `dimB` are set to `sqrt(rho_dim)`.
                                           If `int`, `dimA` is `dim_arg` and `dimB` is `rho_dim / dim_arg`.
                                           If `list`, `dimA` is `dim_arg[0]` and `dimB` is `dim_arg[1]`.
        """
        mat1 = np.random.rand(rho_dim, rho_dim) + 1j * np.random.rand(rho_dim, rho_dim)
        mat1 = mat1 @ mat1.conj().T
        rho = np.divide(mat1, np.trace(mat1))
        result = None
        if dim_arg == None:
            d = int(np.round(np.sqrt(rho_dim)))
            result = benchmark(
                log_negativity,
                rhoAB = rho,
                dimA = d, # dimA is set to sqrt(rho_dim).
                dimB = d # dimB is set to sqrt(rho_dim).
            )
        elif isinstance(dim_arg, int):
            result = benchmark(
                log_negativity,
                rhoAB = rho,
                dimA = dim_arg, # dimA is specified by dim_arg.
                dimB = int(rho_dim/dim_arg) # dimB is calculated from rho_dim and dim_arg.
            )
        elif isinstance(dim_arg, list):
            result = benchmark(
                log_negativity,
                rhoAB = rho,
                dimA=dim_arg[0], # dimA is the first element of dim_arg.
                dimB=dim_arg[1]  # dimB is the second element of dim_arg.
            )
        assert result is not None


class TestVonNeumannEntropyBenchmarks:
    """Benchmarks for the `qutipy.entropies.entropy` function."""
    @pytest.mark.parametrize(
        "dim",
        [4, 16, 32, 64, 128, 256],
        ids = lambda x: str(x)
    )
    def test_bench__von_neumann_entropy__vary__rho(self, benchmark, dim):
        """Benchmark `entropy` by varying the dimension of the density matrix.

        Fixed Parameters:
            - None: All relevant parameters are varied through `pytest.mark.parametrize`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the square density matrix `rho`.
        """
        mat1 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        mat1 = mat1 @ mat1.conj().T
        rho = np.divide(mat1, np.trace(mat1))

        result = benchmark(entropy, rho=rho)

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
            K=kraus_ops
        )
        assert result.shape == (dim**2, dim**2)