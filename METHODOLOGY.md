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
