import pytest
import numpy as np
import itertools

from qutipy.general_functions import partial_trace
from qutipy.states import random_density_matrix
from qutipy.gates import RandomUnitary
from qutipy.general_functions import random_PSD_operator
from qutipy.distance import norm_trace_dist

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