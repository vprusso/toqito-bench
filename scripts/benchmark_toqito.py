import pytest
import numpy as np

from toqito.channels import partial_trace


class TestPartialTraceBenchmarks:
    """Benchmarks for the `toqito.channels.partial_trace` function."""

    @pytest.mark.parametrize(
        "matrix_size",
        [4, 16, 64, 256],
        ids=lambda x: str(x),
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
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )

        result = benchmark(partial_trace, input_mat=input_mat, sys=None, dim=None)

        assert result.shape[0] <= matrix_size

    @pytest.mark.parametrize(
        "sys",
        [
            [0],
            [1],
            [0, 1],
            [0, 2],
        ],
        ids=lambda x: str(x),
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
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )

        if sys == [0, 2]:
            # Assume input_mat is composed of 4 systems with dim = 2 each.
            dim = [2, 2, 2, 2]
        elif sys == [0, 1]:
            # Assume input_mat is composed of 2 systems with dim = 4 each.
            dim = [4, 4]
        else:
            # Assume default behaviour.
            dim = None

        result = benchmark(partial_trace, input_mat=input_mat, sys=sys, dim=dim)

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
        ids=lambda x: str(x),
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

        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
            matrix_size, matrix_size
        )

        result = benchmark(partial_trace, input_mat=input_mat, sys=None, dim=dim)

        assert result is not None
