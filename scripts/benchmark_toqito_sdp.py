"""Benchmarks for toqito's optimization layer.

The existing suite in ``benchmark_toqito.py`` covers array primitives (``basis``,
``swap``, ``pauli``, ``partial_trace``, the ``random_*`` family). Those are thin
wrappers over BLAS, and they are also the only functions with counterparts in
QuTiPy, Ket.jl and QETLAB, so they are what cross-language comparison can reach.

This module covers the semidefinite-programming layer instead: state
discrimination and exclusion, separability testing, and the symmetric extension
hierarchy. No comparison package implements these, so the baseline here is
toqito's own history rather than another library. That makes these benchmarks
the ones that detect a regression in the code that actually dominates runtime.

Sizes were chosen by measurement, not by guesswork. ``state_distinguishability``
costs roughly 0.04 s, 0.31 s, 9.2 s and 234 s at local dimension 8, 16, 32 and
64, so the scaling tests stop at 16 by default; ``pytest-benchmark`` runs each case many
times, and a nine-second call would put a single parameter into the minutes.
Anything heavier is marked ``slow`` and excluded unless asked for.
"""

import numpy as np
import pytest

from toqito.rand import random_density_matrix, random_states
from toqito.state_opt import (
    state_distinguishability,
    state_exclusion,
    symmetric_extension_hierarchy,
)
from toqito.state_props import is_separable
from toqito.states import bell, isotropic, tile


def _fixed_states(num_states: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    """Return a deterministic set of random pure states.

    Benchmarks must be reproducible across runs, so the seed is always pinned.
    """
    return random_states(num_states, dim, seed=seed)


class TestStateDistinguishabilityBenchmarks:
    """Benchmarks for `toqito.state_opt.state_distinguishability`."""

    @pytest.mark.parametrize("dim", [4, 8, 16], ids=lambda x: str(x))
    def test_bench__state_distinguishability__vary__dim(self, benchmark, dim):
        """Benchmark minimum-error discrimination as the local dimension grows.

        This is the primary scaling curve for the module. Cost grows steeply:
        roughly 0.04 s at dim 8 and 0.31 s at dim 16, so dim 32 (about 9 s) is
        left to the `slow` test below.

        Fixed Parameters:
            - `num_states`: 4 states throughout, so only `dim` varies.
            - `measurement`: default "positive" (global POVM).
            - `primal_dual`: default "dual".

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of each pure state.
        """
        states = _fixed_states(4, dim)

        result = benchmark(state_distinguishability, states)

        assert 0.0 <= result[0] <= 1.0 + 1e-6

    @pytest.mark.slow
    def test_bench__state_distinguishability__vary__dim_large(self, benchmark):
        """Benchmark the largest dimension that is still tractable.

        Separated from the main scaling test because a single call costs about
        nine seconds, which `pytest-benchmark` multiplies by its round count.

        Fixed Parameters:
            - `num_states`: 4.
            - `dim`: 32.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
        """
        states = _fixed_states(4, 32)

        result = benchmark.pedantic(state_distinguishability, args=(states,), rounds=1, iterations=1)

        assert 0.0 <= result[0] <= 1.0 + 1e-6

    @pytest.mark.parametrize("num_states", [2, 3, 4, 6], ids=lambda x: str(x))
    def test_bench__state_distinguishability__vary__num_states(self, benchmark, num_states):
        """Benchmark discrimination as the number of states grows.

        The SDP gains one measurement operator per state, so this isolates the
        cost of ensemble size from the cost of dimension.

        Fixed Parameters:
            - `dim`: 8.
            - `measurement`: default "positive".

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            num_states (int): Number of states in the ensemble.
        """
        states = _fixed_states(num_states, 8)

        result = benchmark(state_distinguishability, states)

        assert 0.0 <= result[0] <= 1.0 + 1e-6

    @pytest.mark.parametrize("primal_dual", ["primal", "dual"], ids=lambda x: x)
    def test_bench__state_distinguishability__param__primal_dual(self, benchmark, primal_dual):
        """Compare the primal and dual formulations at equal problem size.

        Both return the same optimal value, so any difference here is purely the
        cost of the formulation, which is the actionable quantity for choosing a
        default.

        Fixed Parameters:
            - `num_states`: 4, `dim`: 8.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            primal_dual (str): Which formulation to solve.
        """
        states = _fixed_states(4, 8)

        result = benchmark(state_distinguishability, states, primal_dual=primal_dual)

        assert 0.0 <= result[0] <= 1.0 + 1e-6

    @pytest.mark.parametrize("strategy", ["min_error", "unambiguous"], ids=lambda x: x)
    def test_bench__state_distinguishability__param__strategy(self, benchmark, strategy):
        """Compare the minimum-error and unambiguous figures of merit.

        Unambiguous discrimination adds an inconclusive outcome, so the SDP
        carries one extra operator and a different constraint set.

        Fixed Parameters:
            - `num_states`: 3, `dim`: 4.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            strategy (str): Figure of merit to optimize.
        """
        states = _fixed_states(3, 4)

        result = benchmark(state_distinguishability, states, strategy=strategy)

        assert 0.0 <= result[0] <= 1.0 + 1e-6


class TestStateExclusionBenchmarks:
    """Benchmarks for `toqito.state_opt.state_exclusion`."""

    @pytest.mark.parametrize("dim", [4, 8, 16], ids=lambda x: str(x))
    def test_bench__state_exclusion__vary__dim(self, benchmark, dim):
        """Benchmark state exclusion as the local dimension grows.

        Mirrors the discrimination scaling test so the two figures of merit can
        be compared on the same axis.

        Fixed Parameters:
            - `num_states`: 4.
            - `primal_dual`: default "dual".

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Dimension of each pure state.
        """
        states = _fixed_states(4, dim)

        result = benchmark(state_exclusion, states)

        assert result[0] >= -1e-6

    @pytest.mark.parametrize("primal_dual", ["primal", "dual"], ids=lambda x: x)
    def test_bench__state_exclusion__param__primal_dual(self, benchmark, primal_dual):
        """Compare the primal and dual formulations for exclusion.

        Fixed Parameters:
            - `num_states`: 4, `dim`: 8.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            primal_dual (str): Which formulation to solve.
        """
        states = _fixed_states(4, 8)

        result = benchmark(state_exclusion, states, primal_dual=primal_dual)

        assert result[0] >= -1e-6


class TestIsSeparableBenchmarks:
    """Benchmarks for `toqito.state_props.is_separable`.

    `is_separable` runs a chain of criteria and returns on the first verdict, so
    its cost is governed by *which* criterion fires rather than by dimension. A
    Haar-random density matrix is rejected by the PPT test in well under a
    millisecond and never reaches the expensive stages, which makes a
    dimension-only benchmark misleading. These cases are therefore chosen to
    land on different points in the chain.
    """

    @staticmethod
    def _tile_bound_entangled() -> np.ndarray:
        """Return the 3x3 bound entangled state built from the tile UPB.

        Being PPT, it survives the cheap criteria and forces the later, more
        expensive tests to run.
        """
        proj = sum(t @ t.conj().T for t in (tile(i) for i in range(5)))
        rho = np.eye(9) - proj
        return rho / np.trace(rho)

    @pytest.mark.parametrize(
        "state_name",
        ["max_mixed", "bell", "isotropic_npt", "random_npt", "tile_bound_entangled"],
        ids=lambda x: x,
    )
    def test_bench__is_separable__param__criterion(self, benchmark, state_name):
        """Benchmark separability testing across states that trigger different criteria.

        The cases fire, in order: the Gurvits-Barnum ball, the pure-state Schmidt
        rank test, the PPT criterion (twice), and finally the bound entangled
        case that no cheap criterion can settle.

        Fixed Parameters:
            - `dim`: implied by each state, either [2, 2] or [3, 3].

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            state_name (str): Which state family to test.
        """
        states = {
            "max_mixed": (np.eye(4) / 4, [2, 2]),
            "bell": (bell(0) @ bell(0).conj().T, [2, 2]),
            "isotropic_npt": (isotropic(3, 0.9), [3, 3]),
            "random_npt": (random_density_matrix(9, seed=2), [3, 3]),
            "tile_bound_entangled": (self._tile_bound_entangled(), [3, 3]),
        }
        rho, dim = states[state_name]

        result = benchmark(is_separable, rho, dim=dim)

        assert isinstance(result[0], (bool, np.bool_))

    @pytest.mark.parametrize("dim", [2, 3, 4], ids=lambda x: str(x))
    def test_bench__is_separable__vary__dim(self, benchmark, dim):
        """Benchmark separability testing on a maximally mixed state as dimension grows.

        The maximally mixed state is separable and is caught by an early
        criterion in every dimension, so this isolates the fixed overhead of the
        dispatch chain from the cost of any single criterion.

        Fixed Parameters:
            - state: maximally mixed on a `dim` x `dim` bipartite system.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            dim (int): Local dimension of each subsystem.
        """
        rho = np.eye(dim * dim) / (dim * dim)

        result = benchmark(is_separable, rho, dim=[dim, dim])

        assert isinstance(result[0], (bool, np.bool_))


class TestSymmetricExtensionHierarchyBenchmarks:
    """Benchmarks for `toqito.state_opt.symmetric_extension_hierarchy`."""

    @pytest.mark.parametrize("level", [1, 2], ids=lambda x: str(x))
    def test_bench__symmetric_extension_hierarchy__vary__level(self, benchmark, level):
        """Benchmark the hierarchy as the extension level grows.

        Level is the dominant cost parameter: the SDP variable is extended by one
        copy of the second subsystem per level, so the matrix dimension grows
        geometrically. Measured at roughly 0.16 s and 0.76 s for levels 1 and 2,
        which is why level 3 is omitted.

        Fixed Parameters:
            - `states`: two Bell states.

        Args:
            benchmark (pytest_benchmark.fixture.BenchmarkFixture): The pytest-benchmark fixture.
            level (int): Level of the symmetric extension hierarchy.
        """
        states = [bell(0) @ bell(0).conj().T, bell(1) @ bell(1).conj().T]

        result = benchmark(symmetric_extension_hierarchy, states=states, level=level)

        assert 0.0 <= result <= 1.0 + 1e-6
