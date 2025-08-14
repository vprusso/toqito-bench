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


# ---- random_unitary ---- # 
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


# ---- random_povm ---- # 
SUITE["TestRandomPOVMBenchmarks"] = BenchmarkGroup()
SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__dim"] = BenchmarkGroup()
SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__num_inputs"] = BenchmarkGroup()
SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__num_outputs"] = BenchmarkGroup()
SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__dim_num_inputs_num_outputs"] = BenchmarkGroup()

"""Benchmark `random_povm` with varying POVM dimensions."""
dim_group = SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__dim"]


for dim in [4, 16, 64, 256]
    key = "test_bench__random_povm__vary__dim[$dim]"
    num_inputs = 4
    num_outputs = 4
    dim_group[key] = @benchmarkable begin
        povms = [random_povm($dim, $num_outputs, $dim) for _ in 1:$num_inputs]
        povm_array = Array{ComplexF64, 4}(undef, $dim, $dim, $num_inputs, $num_outputs)
        for i in 1:$num_inputs, j in 1:$num_outputs
            povm_array[:, :, i, j] = povms[i][j].data
        end
        @assert size(povm_array) == ($dim, $dim, $num_inputs, $num_outputs)
    end
end

"""Benchmark `random_povm` with varying number of measurement inputs."""
input_group = SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__num_inputs"]
for num_inputs in [4, 16, 64, 256]
    dim = 4
    num_outputs = 4
    key = "test_bench__random_povm__vary__num_inputs[$num_inputs]"
    input_group[key] = @benchmarkable begin
        povms = [random_povm($dim, $num_outputs, $dim) for _ in 1:$num_inputs]
        povm_array = Array{ComplexF64, 4}(undef, $dim, $dim, $num_inputs, $num_outputs)
        for i in 1:$num_inputs, j in 1:$num_outputs
            povm_array[:, :, i, j] = povms[i][j].data
        end
        @assert size(povm_array) == ($dim, $dim, $num_inputs, $num_outputs)
    end
end

"""Benchmark `random_povm` with varying number of measurement outputs."""
outputs_group = SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__num_outputs"]
for num_outputs in [4, 16, 64, 256]
    dim = 4
    num_inputs = 4
    key = "test_bench__random_povm__vary__num_outputs[$num_outputs]"
    outputs_group[key] = @benchmarkable begin
        povms = [random_povm($dim, $num_outputs, $dim) for _ in 1:$num_inputs]
        povm_array = Array{ComplexF64, 4}(undef, $dim, $dim, $num_inputs, $num_outputs)
        for i in 1:$num_inputs, j in 1:$num_outputs
            povm_array[:, :, i, j] = povms[i][j].data
        end
        @assert size(povm_array) == ($dim, $dim, $num_inputs, $num_outputs)
    end
end

"""Benchmark `random_povm` with varying combinations of dimensions, inputs, and outputs."""
combo_group = SUITE["TestRandomPOVMBenchmarks"]["test_bench__random_povm__vary__dim_num_inputs_num_outputs"]
for (dim, num_inputs, num_outputs) in [
        (4, 4, 4), (4, 8, 8), (8, 4, 8), (8, 8, 4), (8, 8, 8), (16, 16, 16)
    ]
    key = "test_bench__random_povm__vary__dim_num_inputs_num_outputs[$dim-$num_inputs-$num_outputs]"
    combo_group[key] = @benchmarkable begin
        povms = [random_povm($dim, $num_outputs, $dim) for _ in 1:$num_inputs]
        povm_array = Array{ComplexF64, 4}(undef, $dim, $dim, $num_inputs, $num_outputs)
        for i in 1:$num_inputs, j in 1:$num_outputs
            povm_array[:, :, i, j] = povms[i][j].data
        end
        @assert size(povm_array) == ($dim, $dim, $num_inputs, $num_outputs)
    end
end


# ---- trace_norm ---- # 
SUITE["TestTraceNormBenchmarks"] = BenchmarkGroup()
SUITE["TestTraceNormBenchmarks"]["test_bench__random_povm__vary__dim"] = BenchmarkGroup()

"""Benchmark `trace_norm` with varying matrix dimensions and square/non-square shapes."""
dim_group = SUITE["TestTraceNormBenchmarks"]["test_bench__trace_norm__vary__rho"]
dim_cases = [(4, "square"),(16, "square"),(64, "square"),(128, "square"),(25, "not_square"),(100, "not_square")]

for (dim, is_square) in dim_cases
    key = "test_bench__trace_norm__vary__rho[$dim-$is_square]"
    if is_square == "square"
        dim_group[key] = @benchmarkable begin
            rho = randn(ComplexF64, $dim, $dim)
            val = trace_norm(rho)
            @assert !isnothing(val)
        end
    else
        dim_group[key] = @benchmarkable begin
            rho = randn(ComplexF64, $dim, 2*$dim)
            val = trace_norm(rho)
            @assert !isnothing(val)
        end
    end
end

# ---- entropy ---- # 
SUITE["TestVonNeumannEntropyBenchmarks"] = BenchmarkGroup()
SUITE["TestVonNeumannEntropyBenchmarks"]["test_bench__von_neumann_entropy__vary__rho"] = BenchmarkGroup()

"""Benchmark `von_neumann_entropy` by varying the dimension of the density matrix."""
rho_group = SUITE["TestVonNeumannEntropyBenchmarks"]["test_bench__von_neumann_entropy__vary__rho"]
dims = [4, 16, 32, 64, 128, 256]

for dim in dims
    key = "test_bench__von_neumann_entropy__vary__rho[$dim]"

    rho_group[key] = @benchmarkable begin
        input_mat = randn(ComplexF64, $dim, $dim)
        input_mat = input_mat * adjoint(input_mat)
        rho = input_mat / tr(input_mat)
        h = entropy(rho)
        @assert isa(h, Number) && h >= 0
    end
end

# ----channel_amplitude_damping_generalized---- #
SUITE["TestAmplitudeDampingBenchmarks"] = BenchmarkGroup()
SUITE["TestAmplitudeDampingBenchmarks"]["test_bench__amplitude_damping__vary__input_mat_gamma_prob"] = BenchmarkGroup()

"""Benchmark `amplitude_damping` with varying input matrix presence, damping rate, and probability."""
bench_group = SUITE["TestAmplitudeDampingBenchmarks"]["test_bench__amplitude_damping__vary__input_mat_gamma_prob"]

configs = [
    # input_mat present, gamma, prob
    ("False", 0.0, 0.0),
    ("False", 0.1, 0.5),
    ("False", 0.5, 0.5),
    ("False", 0.7, 0.2),
    ("False", 0.1, 1.0),
    ("False", 0.7, 1.0),
    ("False", 1.0, 1.0),
    ("True", 0.0, 0.0),
    ("True", 0.1, 0.5),
    ("True", 0.5, 0.5),
    ("True", 0.7, 0.0),
    ("True", 1.0, 1.0)
]

for (input_mat, gamma, prob) in configs
    key = "test_bench__amplitude_damping__vary__input_mat_gamma_prob[$input_mat-$gamma-$prob]"
    if input_mat == "True"
        mat = randn(2, 2) + im * randn(2, 2)
        rho = mat * adjoint(mat)
        rho = rho / tr(rho)
        bench_group[key] = @benchmarkable begin
            kraus_ops = channel_amplitude_damping_generalized($prob, $gamma)
            result = sum(K * $rho * K' for K in kraus_ops)
            @assert isapprox(tr(result), 1.0; atol=1e-10)
        end
    else
        bench_group[key] = @benchmarkable begin
            kraus_ops = channel_amplitude_damping_generalized($prob, $gamma)
            @assert length(kraus_ops) == 4
        end
    end
end


# ---- channel_bit_flip ---- # 
SUITE["TestBitflipBenchmarks"] = BenchmarkGroup()
SUITE["TestBitflipBenchmarks"]["test_bench__bitflip__vary__input_mat_prob"] = BenchmarkGroup()

"""Benchmark `channel_bit_flip` with varying input matrix presence and bitflip probability."""
prob_group = SUITE["TestBitflipBenchmarks"]["test_bench__bitflip__vary__input_mat_prob"]

cases = [
    (false, 0.0),
    (false, 0.2),
    (false, 0.8),
    (false, 1.0),
    (true,  0.0),
    (true,  0.2),
    (true,  0.8),
    (true,  1.0),
]

for (input_mat, prob) in cases
    key = "test_bench__bitflip__vary__input_mat_prob[$(uppercasefirst(string(input_mat)))-$prob]"

    if input_mat
        prob_group[key] = @benchmarkable begin
            A = randn(ComplexF64, 2, 2)
            rho = A * adjoint(A)
            rho = rho / tr(rho)
            kraus_ops = channel_bit_flip(1 - $prob)
            out = zero(rho)
            for K in kraus_ops
                out += K * rho * adjoint(K)
            end
            @assert isapprox(tr(out), 1.0; atol=1e-10)
        end
    else
        prob_group[key] = @benchmarkable begin
            kraus_ops = channel_bit_flip(1 - $prob)
            @assert length(kraus_ops) == 2
        end
    end
end


# ---- ket ---- # 
SUITE["TestBasisBenchmarks"] = BenchmarkGroup()
SUITE["TestBasisBenchmarks"]["test_bench__basis__vary__dim"] = BenchmarkGroup()

dim_group = SUITE["TestBasisBenchmarks"]["test_bench__basis__vary__dim"]

for dim in [4, 16, 64, 256]
    key = "test_bench__basis__vary__dim[$dim]"
    dim_group[key] = @benchmarkable begin
        v = ket(Float64, 3, $dim)
        @assert length(v) == $dim
    end
end


# ---- permute_systems---- #
SUITE["TestPermuteSystemBenchmarks"] = BenchmarkGroup()
SUITE["TestPermuteSystemBenchmarks"]["test_bench__permute_systems__vary__dim_perm"] = BenchmarkGroup()
SUITE["TestPermuteSystemBenchmarks"]["test_bench__permute_systems__vary__vector_input"] = BenchmarkGroup()

dim_perm_group = SUITE["TestPermuteSystemBenchmarks"]["test_bench__permute_systems__vary__dim_perm"]
dim_perm_cases = [
    ([2, 2], [1, 0]),
    ([4, 4], [1, 0]),
    ([8, 8], [1, 0]),
    ([2, 2, 2], [1, 2, 0]),
    ([4, 4, 4], [2, 0, 1]),
    ([2, 2, 2, 2], [3, 2, 1, 0]),
    ([4, 4, 4, 4], [3, 0, 2, 1]),
]

for (dims, perm) in dim_perm_cases
    size = prod(dims)
    perm_jl = [p+1 for p in perm]
    key = "test_bench__permute_systems__vary__dim_perm[$dims-$perm]"
    dim_perm_group[key] = @benchmarkable begin
        mat = randn(ComplexF64, $size, $size)
        result = permute_systems(mat, $perm_jl, $dims)
    end
end
    
vec_group = SUITE["TestPermuteSystemBenchmarks"]["test_bench__permute_systems__vary__vector_input"]

vec_cases = [
    (4, [2, 2], [1, 0]),
    (8, [2, 2, 2], [1, 2, 0]),
    (16, [4, 4], [1, 0]),
    (64, [4, 4, 4], [2, 1, 0]),
    (256, [16, 16], [1, 0]),
]

for (size, dims, perm) in vec_cases
    perm_jl = [p+1 for p in perm]
    #{'size': 16, 'dim': [4, 4], 'perm': [1, 0]}
    key = "test_bench__permute_systems__vary__vector_input[{'size': $size, 'dim': $dims, 'perm': $perm}]"
    vec_group[key] = @benchmarkable begin
        vec = randn(ComplexF64, $size)
        result = permute_systems(vec, $perm_jl, $dims)
        @assert length(result) == $size
    end
end

# ---- choi---- #
SUITE["TestKraustoChoiBenchmarks"] = BenchmarkGroup()
SUITE["TestKraustoChoiBenchmarks"]["test_bench__choi_representation__vary__kraus_ops"] = BenchmarkGroup()
SUITE["TestKraustoChoiBenchmarks"]["test_bench__choi_representation__param__sys"] = BenchmarkGroup()

choi_group = SUITE["TestKraustoChoiBenchmarks"]["test_bench__choi_representation__vary__kraus_ops"]

# (dim, num_ops, cp)
cases = [
    (2, 4, true),
    (2, 32, true),
    (4, 4, true),
    (4, 32, true),
    (16, 4, true),
    (16, 32, true),
    # non-CP cases are NOT supported in ketjl.
]

for (dim, num_ops, cp) in cases
    key = "test_bench__choi_representation__vary__kraus_ops[$dim-$num_ops-True]"
    if cp
        choi_group[key] = @benchmarkable begin
            K = [randn(ComplexF64, $dim, $dim) for _ in 1:$num_ops]
            C = choi(K)
            @assert size(C) == ($dim^2, $dim^2)
        end
    end
end

sys_group = SUITE["TestKraustoChoiBenchmarks"]["test_bench__choi_representation__param__sys"]

# Only sys==2 is supported,
key = "test_bench__choi_representation__param__sys[2]"
sys_group[key] = @benchmarkable begin
    dim = 16
    num_ops = 4
    K = [randn(ComplexF64, dim, dim) for _ in 1:num_ops]
    C = choi(K)
    @assert size(C) == (dim^2, dim^2)
end

# ---- pauli ---- #
SUITE["TestPauliBenchmarks"] = BenchmarkGroup()
SUITE["TestPauliBenchmarks"]["test_bench__pauli__vary__ind"] = BenchmarkGroup()

pauli_group = SUITE["TestPauliBenchmarks"]["test_bench__pauli__vary__ind"]

for j in 0:3
    key1 = "test_bench__pauli__vary__ind[int-$j]"
    pauli_group[key1] = @benchmarkable begin
        M = pauli($j)
        @assert size(M) == (2,2)
    end
end

for n in (2, 4, 8)
    key2 = "test_bench__pauli__vary__ind[list-$n]"
    inds = rand(0:3, n)
    pauli_group[key2] = @benchmarkable begin
        M = pauli($inds)
        @assert size(M) == (2^$n, 2^$n)
    end
end

# ---- permutation_matrix---- #
SUITE["TestPermutationOperatorBenchmarks"] = BenchmarkGroup()

dim_perm_group = SUITE["TestPermutationOperatorBenchmarks"]["test_bench__permutation_operator__vary__dim_perm"] = BenchmarkGroup()

cases = [
    # 2 subsystems
    (2, [1, 0]),  # 4x4 matrix (swap)
    (16, [1, 0]), # 256x256 matrix (swap)

    # 3 subsystems
    (2, [1, 2, 0]), # 8x8 matrix (cycle)

    # 4 subsystems
    (2, [1, 2, 0, 3]), # 16x16 matrix

    # 5 subsystems
    (2, [1, 2, 0, 3, 4]), # 32x32 matrix

    # 6 subsystems
    (2, [1, 2, 5, 0, 3, 4]), # 64x64 matrix (complex shuffle)
    
    # (16, [1, 2, 0]),
    # (8, [1, 2, 0, 3]),
]

for (dim, perm) in cases
    perm_jl = [p+1 for p in perm]
    local key = "test_bench__permutation_operator__vary__dim_perm[$dim-$perm]"
    dim_perm_group[key] = @benchmarkable begin
        p = permutation_matrix(ComplexF64, $dim, $perm_jl)
        @assert size(p) == ($dim^(length($perm_jl)), $dim^(length($perm_jl)))
    end
end


# ---- apply_map ---- #

# SUITE["TestApplyChannelBenchmarks"] = BenchmarkGroup()
# SUITE["TestApplyChannelBenchmarks"]["test_bench__apply_channel__vary__phi_op"] = BenchmarkGroup()

# phi_op_group = SUITE["TestApplyChannelBenchmarks"]["test_bench__apply_channel__vary__phi_op"]
# cases = [
#     ("kraus", 4),
#     ("kraus", 16),
#     ("kraus", 64),
#     ("kraus", 256),
#     # ("choi", 2),
#     # ("choi", 4),
#     # ("choi", 16),
#     # ("choi", 64),
#     # ("choi", (4, 16)),
#     # ("choi", (32, 64)),
# ]

# for (phi_op_type, dim) in cases
#     key = "test_bench__applymap__vary__phi_op[$phi_op_type-$(dim)]"
#     if phi_op_type == "kraus"
#         d = dim
#         input_mat = randn(ComplexF64, d, d)
#         K = [randn(ComplexF64, d, d) for _ in 1:4]
#         phi_op_group[key] = @benchmarkable begin
#             result = applymap(K, $input_mat)
#             @assert size(result) == ($d, $d)
#         end
#     end
# end