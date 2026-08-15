# Methodology

## Benchmark Test Naming Conventions

To ensure consistency and enable automated result parsing, all 
benchmark test functions **must** follow this naming format:

`test_bench__[function]__[test_type]__[params]`

### Components

- `test_bench`: Prefix for all benchmark tests.
- `[function]`: Name of the function being benchmarked (e.g.,`random_density_matrix`).
- `[test_type]`: 
  - Use `vary` for scaling tests.
  - Use `param` for fixed-feature tests.
- `[params]`: Parameter(s) being tested (e.g., `dim`, `is_real`, `distance_metric`). For multiple parameters, separate with underscores.

### Examples

| Purpose                        | Function Name                                               |
|-------------------------------|--------------------------------------------------------------|
| Varying `dim`                 | `test_bench__random_density_matrix__vary__dim`                  |
| Varying `dim` and `k_param`  | `test_bench__random_density_matrix__vary__dim_kparam`            |
| Testing fixed `is_real=True` | `test_bench__random_density_matrix__param__is_real`              |
| Testing `distance_metric`    | `test_bench__random_density_matrix__param__distance_metric`      |

## Run conditions

Benchmark numbers are only meaningful if the machine is quiet. Both toqito and
the comparison packages call into a threaded BLAS, so a loaded machine does not
slow everything down uniformly, it distorts the *ratios* between libraries
depending on how each one threads.

The failure is easy to miss because the numbers still look plausible. On a
12-core machine at load average 7.9, a bare `numpy.linalg.qr` of a 256 x 256
matrix measured 297 ms, against single-digit milliseconds for the same call on
an idle machine. Any comparison drawn from that run would have attributed the
difference to the library under test.

Before recording results:

- check `uptime`; the load average should be near zero, not merely below the
  core count,
- pin BLAS threading (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) so thread
  scheduling is not part of what is being measured,
- sanity-check one primitive of known cost, such as a `qr` of a 256 x 256
  matrix, and discard the run if it is far from expectation.

## Comparing across different data representations

toqito returns dense NumPy arrays. QuTiP returns `Qobj`, whose backing store may
be sparse (CSR) or dense, chosen per operation. When QuTiP can stay sparse and
toqito must materialise a dense array, the two are not doing the same work and
the ratio measures representation rather than implementation.

`ket2dm` on a basis vector of dimension 256 is the clearest case. QuTiP returns
a CSR `Qobj` in about 7 us; toqito builds the full 256 x 256 array in about
48 us, which looks like a 7x deficit. Forcing QuTiP to materialise the same dense
array with `.full()` costs about 27 us, so the like-for-like gap is closer to
1.7x. toqito's implementation is already at `numpy.outer` speed and faster than a
bare `v @ v.conj().T`.

Before recording any comparison, check what each side actually returns. Either
force both to the same representation, or exclude the operation and say why.
Operations where both sides are dense by construction (random state and unitary
generation, distances and entropies on dense inputs) do not have this problem.
