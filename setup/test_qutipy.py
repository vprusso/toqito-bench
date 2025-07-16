import numpy as np


def qutipy_ket_benchmark():
    """Create basis vectors using QuTIpy for benchmarking."""
    from qutipy.general_functions import ket

    return ket(2, 0)


def qutipy_tensor_product_benchmark():
    """Benchmark tensor product of basis vectors."""
    from qutipy.general_functions import ket

    return ket(2, [0, 0])


def qutipy_random_state_benchmark(dim=4):
    """Generate random quantum states for benchmarking."""
    try:
        from qutipy.states import random_state_vector

        return random_state_vector(dim)
    except ImportError:
        print("⚠️  Using fallback random state generation")
        state = np.random.rand(dim) + 1j * np.random.rand(dim)
        return state / np.linalg.norm(state)


def qutipy_random_density_benchmark(dim=4):
    """Generate random density matrix for benchmarking."""
    try:
        from qutipy.states import random_density_matrix

        matrix = random_density_matrix(dim)
        return matrix
    except ImportError:
        print("⚠️  Using fallback density matrix generation")
        state = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        state = state @ state.conj().T
        return state / np.trace(state)


def test_qutipy_environment_setup():
    """Test that QuTIpy and required packages are available."""
    packages = ["qutipy", "numpy", "scipy", "cvxpy", "sympy", "pytest"]
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
    print("=== QuTIpy Environment Test ===")
    test_qutipy_environment_setup()

    print("\n=== Demo QuTIpy Operations ===")

    try:
        # Test basis vector creation
        ket_state = qutipy_ket_benchmark()
        print(f"✓ Basis vector created with shape: {ket_state.shape}")

        # Test tensor product
        tensor_state = qutipy_tensor_product_benchmark()
        print(f"✓ Tensor product created with shape: {tensor_state.shape}")

        # Test random state generation
        random_state = qutipy_random_state_benchmark(4)
        print(f"✓ Random state generated with shape: {random_state.shape}")

        # Test random density matrix
        random_density = qutipy_random_density_benchmark(4)
        print(f"✓ Random density matrix generated with shape: {random_density.shape}")

    except Exception as e:
        print(f"⚠️  Error during operations: {e}")
        print("This might be due to QuTIpy version differences")

    print("\nQuTIpy environment setup complete!")

    print("\nTo run benchmarks, use one of the following commands:\n")

    print("1. Run all Toqito benchmarks using make:")
    print("   make benchmark-qutipy-full\n")

    print("2. Run any specific file manually (from the project root):")
    print(
        "   cd env/qutipy-env && poetry run python ../../path/to/your/file/from/root/directory\n"
    )


if __name__ == "__main__":
    main()
