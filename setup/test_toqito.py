import pytest
import numpy as np


def toqito_bell_state_benchmark():
    """Create Bell states using toqito for benchmarking."""
    from toqito.states import bell
    return bell(0)


def toqito_entanglement_measure():
    """Measure entanglement using toqito concurrence."""
    from toqito.states import bell
    from toqito.state_props import concurrence
    rho = bell(0) @ bell(0).conj().T
    return concurrence(rho)


def toqito_random_state_generation(dim=4):
    """Generate random quantum states for benchmarking."""
    from toqito.rand import random_state_vector
    return random_state_vector(dim)




class TestToqitoBenchmarks:
    """Test class containing toqito benchmark tests."""
    
    def test_bell_state_benchmark(self, benchmark):
        """Benchmark Bell state creation."""
        result = benchmark(toqito_bell_state_benchmark)
        assert result.shape == (4, 1)
    
    def test_entanglement_benchmark(self, benchmark):
        """Benchmark entanglement measurement."""
        result = benchmark(toqito_entanglement_measure)
        assert 0 <= result <= 1
    
    def test_random_state_benchmark(self, benchmark):
        """Benchmark random state generation."""
        result = benchmark(toqito_random_state_generation, 8)
        assert result.shape == (8, 1)

def test_toqito_environment_setup():
    """Test that toqito and required packages are available."""
    packages = ['toqito', 'numpy', 'pytest']
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} failed to import")
    
    assert len(missing) == 0, f"Missing packages: {missing}"


def main():
    """Main function for direct execution."""
    print("=== Toqito Environment Test ===")
    test_toqito_environment_setup()
    
    print("\n=== Demo Toqito Operations ===")
    
    # Test Bell state creation
    bell_state = toqito_bell_state_benchmark()
    print(f"✓ Bell state created with shape: {bell_state.shape}")
    
    # Test entanglement measure
    concurrence_val = toqito_entanglement_measure()
    print(f"✓ Bell state concurrence: {concurrence_val:.4f}")
    
    # Test random state generation
    random_state = toqito_random_state_generation(4)
    print(f"✓ Random state generated with shape: {random_state.shape}")
    
    
    print("\nToqito environment setup successful!")
    print("\nTo run benchmarks, use one of these commands:")
    print("\n2. Using make:")
    print("   make test-toqito-benchmarks")
    print("\n3. Direct pytest:")
    print("   pytest benchmarks/benchmark_toqito.py --benchmark-only")


if __name__ == "__main__":
    main()
