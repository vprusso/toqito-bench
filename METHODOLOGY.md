# Methodology

## Benchmark Test Naming Conventions

To ensure consistency and enable automated result parsing, all 
benchmark test functions **must** follow this naming format:

`test_bench_[function]_[test_type]_[params]`

### Components

- `test_bench_`: Prefix for all benchmark tests.
- `[function]`: Name of the function being benchmarked (e.g.,`random_density_matrix`).
- `[test_type]`: 
  - Use `vary` for scaling tests.
  - Use `param` for fixed-feature tests.
- `[params]`: Parameter(s) being tested (e.g., `dim`, `is_real`, `distance_metric`). For multiple parameters, separate with underscores.

### Examples

| Purpose                        | Function Name                                               |
|-------------------------------|--------------------------------------------------------------|
| Varying `dim`                 | `test_bench_random_density_matrix_vary_dim`                  |
| Varying `dim` and `k_param`  | `test_bench_random_density_matrix_vary_dim_kparam`            |
| Testing fixed `is_real=True` | `test_bench_random_density_matrix_param_is_real`              |
| Testing `distance_metric`    | `test_bench_random_density_matrix_param_distance_metric`      |
