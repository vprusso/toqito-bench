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
from qutipy.channels import amplitude_damping_channel
from qutipy.channels import bit_flip_channel
from qutipy.channels import dephasing_channel
from qutipy.channels import choi_representation
from qutipy.general_functions import ket
from qutipy.general_functions import SWAP
from qutipy.general_functions import syspermute
from qutipy.general_functions import dag
from qutipy.channels import choi_representation
from qutipy.pauli import generate_nQubit_Pauli
from qutipy.general_functions import permute_tensor_factors
from qutipy.channels import apply_channel

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

class TestAmplitudeDampingBenchmarks:
    """Benchmarks for the `qutipy.channels.amplitude_damping` function."""

    @pytest.mark.parametrize(
        "input_mat, gamma, prob",
        [
            (False, 0.1, 1.0),
            (False, 0.7, 1.0),
            (False, 1.0, 1.0),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__amplitude_damping__vary__input_mat_gamma_prob(self, benchmark, input_mat, gamma, prob):
        """Benchmark `amplitude_damping_channel` with varying damping rate.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            gamma (float): The damping rate for the amplitude damping channel.
        """
        result = benchmark(
                amplitude_damping_channel,
                gamma=gamma
            )
        assert len(result) == 2

class TestBitflipBenchmarks:
    """Benchmarks for the `qutipy.channels.bit_flip_channel` function."""

    @pytest.mark.parametrize(
        "input_mat, prob",
        [
            (False,  0.0),
            (False, 0.2),
            (False, 0.8),
            (False, 1.0),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__bitflip__vary__input_mat_prob(self, benchmark, input_mat, prob):
        """Benchmark `bit_flip_channel` with varying bitflip probability.

        Fixed Parameters:
            - `None`

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            prob (float): The probability `p` used to define the bit-flip channel.
        """
        result = benchmark(bit_flip_channel, p = prob)
        
        assert len(result) == 3

class TestDephasingBenchmarks:
    """Benchmarks for `qutipy.channels.dephasing_channel` function."""
    @staticmethod
    def choi_dephasing(p_vec, dim):
        kraus_ops = dephasing_channel(p_vec, d=dim)
        if dim == 2:
            kraus_ops = kraus_ops[0]
        return choi_representation(K=kraus_ops, dA=dim)

    @pytest.mark.parametrize(
        "dim",
        [2, 4, 8, 16, 32],
        ids = lambda x: str(x)
    )
    def test_bench__dephasing__vary__dim(self, benchmark, dim):
        """Benchmark `dephasing_channel` (via its Choi representation) with varying dimensionality.

        Fixed Parameters:
            - `p`: The probability parameter `p` for the `dephasing_channel` is set dynamically.
                   For `dim=2`, `p=0.5` is used, corresponding to the standard qubit dephasing.
                   For `dim > 2`, `p` is a vector `np.full(dim, 1/dim)`\.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimensionality of the channel.
        """
        p = 0.5 if dim ==2 else np.full(dim, 1/dim)

        # Benchmark the choi_dephasing static method, which in turn calls dephasing_channel.
        choi_mat = benchmark(self.choi_dephasing, p, dim=dim)
        
        assert choi_mat.shape == (dim**2, dim**2)
            
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
        p_vec = np.full(dim, (1 - param_p) / dim)
        p_vec[0] = (1 + (dim - 1) * param_p) / dim
        
        choi_mat = benchmark(self.choi_dephasing, p_vec, dim=dim)
        assert choi_mat.shape == (dim**2, dim**2)

class TestBasisBenchmarks:
    """Benchmarks for the `qutipy.general_functions.ket` function."""

    @pytest.mark.parametrize(
        "dim",
        [4, 16, 64, 256],
        ids = lambda x: str(x),
    )
    def test_bench__basis__vary__dim(self, benchmark, dim):
        """Benchmark `ket` with varying dimension `dim` for a single basis vector.

        Fixed Parameters:
            - `args`: Set to `2` to represent the index of the basis vector (e.g., |2>).

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the basis vector.
        """
        result = benchmark(ket, dim, 2)

        assert result.shape[0] == dim

class TestSwapOperatorBenchmarks:
    """Benchmarks for the `qutipy.general_functions.SWAP` function."""

    @pytest.mark.parametrize(
        "dim, is_sparse",
        [
            (8, False),
            ([4, 4], False),
            ([2, 2, 2], False),
        ],
        ids = lambda x: str(x)
    )
    def test_bench__swap_operator__vary__dim_is_sparse(self, benchmark, dim, is_sparse):

        if isinstance(dim, int):
            dim = [dim, dim]
        
        sys = [1, 2]

        result = benchmark(SWAP, sys=sys, dim=dim)
        expected_size = int(np.prod(dim))
        assert isinstance(result, np.ndarray)
        assert result.shape == (expected_size, expected_size)

class TestPermuteSystemBenchmarks:
    """Benchmarks for the `qutipy.general_functions.syspermute` function."""

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
        """Benchmark `syspermute` with varying subsystem dimensions and permutation orders for mixed states.

        Fixed Parameters:
            - None: All parameters are varied.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (list[int]): A list of the dimensions of the subsystems.
            perm (list[int]): A list containing the desired permutation order (0-indexed).
        """
        matrix_size = int(np.prod(dim))
        input_mat = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)

        # Convert 0-indexed permutation to 1-indexed as required by `syspermute`.
        perm = [p+1 for p in perm]

        result = benchmark(syspermute, X=input_mat, perm=perm, dim=dim)

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
        ids=lambda x: str(x),
    )
    def test_bench__permute_systems__vary__vector_input(self, benchmark, setup):
        """Benchmark `syspermute` with varying input vector sizes, subsystem dimensions, and permutation orders for pure states.

        Fixed Parameters:
            - `X`: A column vector (pure state) generated based on `size`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            setup (dict): A dictionary containing `size` (total dimension of the vector),
                          `dim` (list of subsystem dimensions), and `perm` (permutation order).
        """
        size = setup["size"]

        # Generate a random column vector representing a pure state.
        input_mat = np.random.rand(size, 1) + np.random.rand(size, 1)

        perm = [p + 1 for p in setup["perm"]]

        # Convert 0-indexed permutation to 1-indexed as required by `syspermute`.
        result = benchmark(syspermute, X=input_mat, perm=perm, dim=setup["dim"])

        assert result.shape == (setup["size"], 1)

class TestKraustoChoiBenchmarks:
    """Benchmarks for the qutipy.channels.choi_representation function."""

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
    def test_bench__choi_representation__vary__kraus_ops(self, benchmark, dim, num_ops, cp):
        """Benchmark `choi_representation` by varying the size, number, and CP nature of Kraus operators.

        Fixed Parameters:
            - `adjoint`: Set to `False` as `choi_representation` does not compute the adjoint by default for this representation.
            - `normalized`: Set to `False` to match the unnormalized Choi matrix output.
            - `dA`: Derived directly from `dim`, representing the input dimension of the channel.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of the square Kraus operators, which also sets `dA`.
            num_ops (int): The number of base Kraus operators to generate.
            cp (bool): If `True`, `L` is set to `None` (implying `L=K` for a CP map).
                       If `False`, `L` is set as the conjugate transpose of `K` (for non-CP maps).
        """


        # Generate base Kraus operators for `K`
        base_ops = [np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim) for _ in range(num_ops)]

        K = base_ops
        L = None # Default to None, which implies L=K in choi_representation

        if not cp:
            # For non-completely positive (non-CP) channels, `L` is explicitly set to the dagger of `K`.
            L = [dag(op) for op in base_ops]
        #`dA` is the input dimension, which corresponds to `dim`.
        result = benchmark(choi_representation, K=K, dA=dim, L=L, adjoint=False, normalized=False)
        assert result.shape == (dim**2, dim**2)

    @pytest.mark.parametrize("sys", [2])
    def test_bench__kraus_to_choi__param__sys(self, benchmark, sys):
        """Benchmark `choi_representation` with a fixed `sys` parameter (which is implicitly 2 in qutipy).

        Fixed Parameters:
            - `dim`: Fixed dimension of 16 for Kraus operators and `dA` to represent a moderate size.
            - `num_ops`: Fixed to 4 base Kraus operators to represent a typical number of Kraus operators.
            - `cp`: Fixed to `False` to test a non-completely positive scenario where `L` is distinct from `K`.
            - `adjoint`: Set to `False`.
            - `normalized`: Set to `False`.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            sys (int): Placeholder for the `sys` value. Always 2 for this benchmark due to `qutipy`'s internal implementation.
        """
        
        dim = 16
        num_ops = 4
        K = [np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim) for _ in range(num_ops)]
        L = [dag(op) for op in K]

        result = benchmark(choi_representation, K=K, dA=dim, L=L, adjoint=False, normalized=False)
        assert result.shape == (dim**2, dim**2)

class TestPauliBenchmarks:
    """Benchmarks for the `qutipy.pauli.generate_nQubit_Pauli` function."""

    @pytest.mark.parametrize(
        "type, ind",
        [   
            # Test individual Pauli operators (n=1 qubit).
            *itertools.product(["int"], [0, 1, 2, 3]),
            # Test tensor products of Pauli operators with varying number of qubits
            # (n=2, 4, 8 qubits).
            *itertools.product(["list"], [2, 4, 8]),
        ]
    )
    def test_bench__pauli__vary__ind(self, benchmark, type, ind):
        """Benchmark `generate_nQubit_Pauli` with varying Pauli operator types and number of qubits.

        Fixed Parameters:
            - `alt`: Set to `False` to use the standard Pauli operator generation method (instead of X/Z product method).

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            type (str): Indicates whether `ind` refers to a single integer Pauli index (`"int"`)
                        or the number of qubits for a random Pauli string (`"list"`).
            ind (int): If `type` is `"int"`, this is the index of the single Pauli operator (0-3).
                       If `type` is `"list"`, this is the number of qubits for the tensor product.
        """
        if type == "int":
            # For single qubit Pauli operators, indices is a list containing one element.
            result = benchmark(generate_nQubit_Pauli, indices=[ind], alt=False)
            assert result.shape == (2, 2)
        elif type == "list":
            # Generate random Pauli string of given length
            indices = np.random.randint(0, 4, size=ind).tolist()
            # The dimension of an n-qubit Pauli operator is 2^n x 2^n
            result = benchmark(generate_nQubit_Pauli, indices=indices, alt=False)
            assert result.shape == (2**ind, 2**ind)

class TestPermutationOperatorBenchmarks:
    """Benchmarks for the `qutipy.general_functions.permute_tensor_factors` function."""

    @pytest.mark.parametrize(
        "dim, perm",
        [
            (2, [1, 0]),
            (2, [1, 2, 0]),
            (2, [1, 2, 0, 3]),
            (2, [1, 2, 0, 3, 4]),
            (2, [1, 2, 5, 0, 3, 4]),
            (16, [1, 0]),
            #(16, [1, 2, 0]),
            #(8, [1, 2, 0, 3]),
        ],
        ids=lambda x: str(x),
    )
    def test_bench__permutation_operator__vary__dim_perm(self, benchmark, dim, perm):
        """Benchmark `permute_tensor_factors` by varying the dimension of each factor and the permutation order.

        Fixed Parameters:
            - None.
    
        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): The dimension of each individual tensor factor (e.g., 2 for qubits, 16 for 16-level systems).
            perm (list[int]): A list representing the desired permutation order of the tensor factors.
        """

        dims = [dim] * len(perm)
        result = benchmark(permute_tensor_factors, perm=perm, dims=dims)

        assert result.shape == (dim ** (max(perm) + 1), dim ** (max(perm) + 1))
        assert isinstance(result, np.ndarray)
    
class TestApplyChannelBenchmarks:
    """Benchmarks for the `qutipy.channels.apply_channel` function"""
    @pytest.mark.parametrize(
        "phi_op_type, dim",
        [
            ("kraus", 4),
            ("kraus", 16),
            ("kraus", 64),
            ("kraus", 256),
        ],
        ids=lambda x: str(x),
    )
    def test_bench__apply_channel__vary__phi_op(self, benchmark, phi_op_type, dim):
        """Benchmark `apply_channel` with Kraus representation by varying input dimensions.

        Fixed Parameters:
            - `phi_op_type`: Set to `"kraus"` to benchmark only the Kraus representation.
            - `sys`: Set to `None` to apply the channel across the entire state.
            - `dim`: Set to `None` since `sys` is `None` and full-state channel application does not require subsystem dimensions.
            - `adjoint`: Set to `False` to benchmark the channel as-is, without adjoint transformation.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            phi_op_type (str): Type of channel representation. Only `"kraus"` is currently supported.
            dim (int): The input and output dimension of the quantum state `rho`.
        """
        
        
        if phi_op_type == "kraus":
            input_mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            kraus_ops = [np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim) for _ in range(4)]
            result = benchmark(apply_channel, K=kraus_ops, rho=input_mat ,sys=None, dim=None, adjoint=False)
            assert result.shape == (dim, dim)
