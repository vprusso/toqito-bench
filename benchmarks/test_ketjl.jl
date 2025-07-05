using Test
using Ket
using BenchmarkTools

@testset "Ket.jl & BenchmarkTools basic setup" begin
    # Test that Ket exports a known function
    @test isdefined(Ket, :ket)
    # Generate a simple |0⟩ state and check its type
    zero_ket = ket(1, 2)
    @test isa(zero_ket, AbstractVector)
    @test length(zero_ket) == 2
    @test zero_ket[1] == 1.0
    @test zero_ket[2] == 0.0

    # Test a simple entanglement function
    bell = state_bell(0,0)
    @test isa(bell, AbstractMatrix)
    @test length(bell) == 16

    # Benchmark a simple operation (not a strict test, but ensures BenchmarkTools works)
    bm = @benchmark ket(1, 2)
    @test bm isa BenchmarkTools.Trial
end

println("✅ Ket.jl and BenchmarkTools loaded and basic functionality verified.")
