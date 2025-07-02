using BenchmarkTools
using LinearAlgebra
using Random

function fibonacci(n::Int)
    """Simple fibonacci function for benchmarking."""
    if n <= 1
        return n
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end

function matrix_multiply(size::Int=100)
    """Matrix multiplication for benchmarking."""
    Random.seed!(42)  # For reproducible results
    a = rand(size, size)
    b = rand(size, size)
    return a * b
end

function benchmark_fibonacci()
    """Benchmark fibonacci function."""
    println("Benchmarking fibonacci(10)...")
    result = @benchmark fibonacci(10)
    println("Result: $(fibonacci(10))")
    @assert fibonacci(10) == 55
    display(result)
    return result
end

function benchmark_matrix_multiply()
    """Benchmark matrix multiplication."""
    println("\nBenchmarking matrix multiplication (50x50)...")
    result = @benchmark matrix_multiply(50)
    matrix_result = matrix_multiply(50)
    println("Result shape: $(size(matrix_result))")
    @assert size(matrix_result) == (50, 50)
    display(result)
    return result
end

function test_environment_setup()
    """Test that all required packages are available."""
    packages = ["BenchmarkTools", "LinearAlgebra", "Random"]
    missing = String[]
    
    for package in packages
        try
            eval(Meta.parse("using $package"))
            println("✓ $package imported successfully")
        catch e
            push!(missing, package)
            println("✗ $package failed to import: $e")
        end
    end
    
    @assert length(missing) == 0 "Missing packages: $missing"
end

function run_benchmarks()
    """Run all benchmark tests."""
    println("=== Running Julia Benchmarks ===")
    
    # Run individual benchmarks
    fib_result = benchmark_fibonacci()
    matrix_result = benchmark_matrix_multiply()
    
    println("\n=== Benchmark Summary ===")
    println("Fibonacci(10) median time: $(median(fib_result.times)) ns")
    println("Matrix multiply (50x50) median time: $(median(matrix_result.times)) ns")
end

function main()
    """Main function for direct execution."""
    println("=== Environment Test ===")
    test_environment_setup()
    
    println("\n=== Demo Calculations ===")
    println("Fibonacci(10) = $(fibonacci(10))")
    println("Matrix multiplication (10x10) completed")
    matrix_multiply(10)
    
    println("\nEnvironment setup successful!")
    println("\nTo run benchmarks, use one of these commands:")
    println("1. Activate Julia environment first:")
    println("   julia --project=env/julia-env")
    println("   julia> include(\"benchmarks/test.jl\")")
    println("   julia> run_benchmarks()")
    println("\n2. Or run directly:")
    println("   julia --project=env/julia-env benchmark/test.jl")
    println("\n3. Or from activated environment:")
    println("   julia --project=env/julia-env -e 'include(\"benchmark/test.jl\"); run_benchmarks()'")
end

# Run main if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
