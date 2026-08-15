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
