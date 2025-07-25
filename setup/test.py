import numpy as np


def fibonacci(n):
    """Simple fibonacci function for benchmarking."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def matrix_multiply(size=100):
    """Matrix multiplication for benchmarking numpy operations."""
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.dot(a, b)


class TestBenchmarks:
    """Test class containing benchmark tests."""

    def test_fibonacci_benchmark(self, benchmark):
        """Benchmark fibonacci function."""
        result = benchmark(fibonacci, 10)
        assert result == 55

    def test_matrix_multiply_benchmark(self, benchmark):
        """Benchmark matrix multiplication."""
        result = benchmark(matrix_multiply, 50)
        assert result.shape == (50, 50)


def test_environment_setup():
    """Test that all required packages are available."""
    packages = ["numpy", "pytest"]
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
    print("=== Environment Test ===")
    test_environment_setup()

    print("\n=== Demo Calculations ===")
    print(f"Fibonacci(10) = {fibonacci(10)}")
    print("Matrix multiplication (10x10) completed")
    matrix_multiply(10)

    print("\nEnvironment setup successful!")
    print("\nTo run benchmarks, use one of these commands:")
    print("1. Activate virtual environment first:")
    print("   source env/python-env/bin/activate")
    print("   pytest benchmarks/test.py")
    print("\n2. Or run directly without activation:")
    print("   env/python-env/bin/python -m pytest benchmark/test.py")


if __name__ == "__main__":
    main()
