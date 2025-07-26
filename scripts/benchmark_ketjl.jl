using Ket
using BenchmarkTools
using LinearAlgebra
using BenchmarkTools
using Statistics
using JSON3

SUITE = BenchmarkGroup()


# ----- helper functions ----

function run_and_export_benchmarks(SUITE; key1=nothing, key2=nothing, json_path="benchmarks.json")
    results = []

    function traverse_and_run(group, top_groupname)
        for (k, v) in group
            if v isa BenchmarkGroup
                traverse_and_run(v, top_groupname)
            else
                name = String(k)
                trial = run(v)
                push!(results, summarize_trial(trial, name))
            end
        end
    end

    if key1 !== nothing && key2 !== nothing
        group = SUITE[key1][key2]
        for (k, v) in group
            param_str = match(r"\[.*\]$", String(k))
            name = param_str === nothing ? string(key2) : "$key2$(param_str.match)"
            trial = run(v)
            push!(results, summarize_trial(trial, name))
        end
    elseif key1 !== nothing
        traverse_and_run(SUITE[key1], key1)
    else
        for (gk, grp) in SUITE
            traverse_and_run(grp, gk)
        end
    end

    open(json_path, "w") do io
        JSON3.pretty(io, results)
    end
end


function summarize_trial(trial, name)
    times = trial.times
    iqr = quantile(times, 0.75) - quantile(times, 0.25)
    iqr_outliers = filter(t -> t < quantile(times, 0.25) - 1.5 * iqr || t > quantile(times, 0.75) + 1.5 * iqr, times)
    mu = mean(times)
    sigma = std(times)
    stddev_outliers = filter(t -> abs(t - mu) > 2*sigma, times)

    return Dict(
        "name" => name,
        "min_ns" => minimum(times),
        "max_ns" => maximum(times),
        "mean_ns" => mu,
        "stddev_ns" => sigma,
        "rounds" => length(times),
        "median_ns" => median(times),
        "iqr_ns" => iqr,
        "q1_ns" => quantile(times, 0.25),
        "q2_ns" => quantile(times, 0.75),
        #"iqr_outliers_ns" => iqr_outliers,
        #"stddev_outliers_ns" => stddev_outliers,
        #"outliers_ns" => string(iqr_outliers),
        "ld15iqr_ns" => quantile(times, 0.15),
        "hd15iqr_ns" => quantile(times, 0.85),
        "ops" => length(times),
        "total_s" => sum(times) / 1e9,
        "iterations" => 1,
        "memory_bytes" => trial.memory,
        "allocations" => trial.allocs
    )
end

# ---- partial_trace ---- # 

SUITE["TestPartialTraceBenchmarks"] = BenchmarkGroup()
SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__input_mat"] = BenchmarkGroup()
SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__dim"] = BenchmarkGroup()
SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__sys"] = BenchmarkGroup()


"""Benchmark `partial_trace` by varying subsystem dimensions (`dim`)."""

dim_group = SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__dim"]
dim_cases = [nothing, [2,2,2,2], [2,2], [3,3], [4,4]]
ids = ["None", "[2, 2, 2, 2]", "[2, 2]", "[3, 3]", "[4, 4]"]

for (dim, id) in zip(dim_cases, ids)
    # Defaults to [2, 2] when `nothing` is provided.
    matrix_size = prod(dim === nothing ? [2,2] : dim)
    input_mat = randn(ComplexF64, matrix_size, matrix_size)
    # Always set to 2.
    remove = 2
    key = "test_bench__partial_trace__vary__dim[" * id * "]"
    if dim === nothing
        dim_group[key] = @benchmarkable partial_trace($input_mat, $remove)
    else
        dim_group[key] = @benchmarkable partial_trace($input_mat, $remove, $dim)
    end
end

"""Benchmark `partial_trace` with varying input matrix sizes."""

input_mat_group = SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__input_mat"]
sizes = [4, 16, 64, 256]

for matrix_size in sizes
    mat = rand(ComplexF64, matrix_size, matrix_size)
    d  = Int(sqrt(matrix_size))
    # Always set to 2.
    remove = 2
    # Calculated as [d, d] where d is the sqrt of the matrix size.
    dims = [d, d]

    key = "test_bench__partial_trace__vary__input_mat[$matrix_size]"
    input_mat_group[key]= @benchmarkable partial_trace($mat, $remove, $dims)
end

"""Benchmark `partial_trace` by tracing out different subsystems."""

sys_group = SUITE["TestPartialTraceBenchmarks"]["test_bench__partial_trace__vary__sys"]
sys_list = [[1], [2], [1,2], [1,3]]
ids = ["[0]", "[1]", "[0, 1]", "[0, 2]"]
    
for (sys, id) in zip(sys_list, ids)
    input_mat = randn(ComplexF64, 16, 16)

    if sys == [1, 3]
        dims = [2, 2, 2, 2]
    elseif sys == [1, 2]
        dims = [4, 4]
    else
        dims = nothing
    end

    key = "test_bench__partial_trace__vary__sys[" * id *"]"

    if dims === nothing
        sys_group[key] = @benchmarkable partial_trace($input_mat, $sys)
    else
        sys_group[key] = @benchmarkable partial_trace($input_mat, $sys, $dims)
    end
end


# ---- random_density_matrix ---- # 
SUITE["TestRandomUnitaryBenchmarks"] = BenchmarkGroup()
SUITE["TestRandomUnitaryBenchmarks"]["test_bench__random_unitary__vary__dim"] = BenchmarkGroup()
SUITE["TestRandomUnitaryBenchmarks"]["test_bench__random_unitary__vary__is_real"] = BenchmarkGroup()

"""Benchmark `random_unitary` with varying matrix dimensions."""
dim_group = SUITE["TestRandomUnitaryBenchmarks"]["test_bench__random_unitary__vary__dim"]
dims = [4, 16, 64, 256, 1024]

for dim in dims
    key = "test_bench__random_unitary__vary__dim[$dim]"
    dim_group[key] = @benchmarkable random_unitary($dim)
end

"""Benchmark `random_unitary` for both real and complex-valued matrices."""
type_group = SUITE["TestRandomUnitaryBenchmarks"]["test_bench__random_unitary__vary__is_real"]
types = [Float64, ComplexF64]

for T in types
    label = !(T <: Complex) ? "True" : "False" 
    key = "test_bench__random_unitary__vary__is_real[$label]"
    dim = 64
    type_group[key] = @benchmarkable random_unitary($T, $dim)
end