import pytest
import numpy as np
from qutipy.general_functions import partial_trace
# from qutipy.states import random_density_matrix


class TestPartialTraceBenchmarks:
    """Benchmarks for the `qutipy.general_functions.partial_trace function`"""

    @pytest.mark.parametrize("matrix_size", [4, 16, 64, 256], ids=lambda x: str(x))
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

        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )

        sys = [2]

        result = benchmark(partial_trace, input_mat, sys=sys, dim=dim)

        assert result.shape[0] <= matrix_size

    @pytest.mark.parametrize(
        "sys_param",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
        ],
        ids=lambda x: str(x),
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
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )

        result = benchmark(partial_trace, input_mat, sys=sys, dim=dim)
        assert result is not None

    @pytest.mark.parametrize(
        "dim",
        [[16, 1], [2, 2], [2, 2, 2, 2], [3, 3], [4, 4]],
        ids=["None", "[2, 2]", "[2, 2, 2, 2]", "[3, 3]", "[4, 4]"],
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

        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )
        sys = [2]

        result = benchmark(partial_trace, input_mat, sys=sys, dim=dim)
        assert result is not None


# class TestRandomDensityMatrixBenchmarks:
#     # present at: https://github.com/sumeetkhatri/QuTIpy/blob/master/qutipy/states.py

#     @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024], ids = lambda x: str(x))
#     def test_Random_Density_Matrix_dim(self, benchmark, dim):
#         """benchmark dim."""
#         result = benchmark(random_density_matrix, dim)

#         assert result.shape == (dim, dim)

#     @pytest.mark.parametrize(
#         "dim, k_param",
#         [
#             (4, 1),
#             (4, 2),
#             (16, 1),
#             (16, 8),
#             (64, 1),
#             (64, 32),
#             (256, 1),
#             (256, 128),
#             (1024, 1),
#             (1024, 512)
#         ],
#         ids=lambda x: f"{x}"
#     )
#     def test_Random_Density_Matrix_dim_and_kparam(self, benchmark, dim, k_param):

#         result = benchmark(random_density_matrix, dim, k_param)
#         assert result.shape == (dim, dim)
