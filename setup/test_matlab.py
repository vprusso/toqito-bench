import numpy as np
import matlab.engine


def matlab_partial_trace(
    rho_np: np.ndarray, dims: list[int], subsystems: list[int]
) -> np.ndarray:
    """
    Compute the partial trace of `rho_np` (a NumPy array) over the given subsystems
    with dimensions `dims` using QETLAB's PartialTrace function.
    Returns the reduced density matrix (as a NumPy array).
    """
    eng = matlab.engine.start_matlab()

    # Add QETLAB to MATLAB's path (replace with your actual QETLAB path)
    qetlab_path = r"C:\Users\pravi\Downloads\QETLAB-0.9"
    eng.addpath(eng.genpath(qetlab_path), nargout=0)

    # Convert the NumPy array to MATLAB double
    rho_mat = matlab.double(rho_np.tolist())

    # Convert dims and subsystems to MATLAB arrays
    # Note: MATLAB uses 1-based indexing, so adjust subsystems
    dims_mat = matlab.double(dims)
    subsys_mat = matlab.double(subsystems)

    try:
        rho_red_mat = eng.PartialTrace(rho_mat, subsys_mat, dims_mat)

        # Convert back to NumPy array
        rho_reduced = np.array(rho_red_mat._data).reshape(rho_red_mat.size)

        # Reshape to proper matrix form
        # Calculate the dimension of the reduced system
        traced_dims = [dims[i - 1] for i in subsystems]  # Adjust for 0-based indexing
        remaining_dim = int(np.prod(dims) / np.prod(traced_dims))
        rho_reduced = rho_reduced.reshape((remaining_dim, remaining_dim))

    except Exception as e:
        print(f"Error calling PartialTrace: {e}")
        eng.quit()
        raise

    eng.quit()
    return rho_reduced


def test_qetlab_installation():
    """Test if QETLAB is properly installed and accessible."""
    try:
        eng = matlab.engine.start_matlab()

        qetlab_path = r"C:\Users\pravi\Downloads\QETLAB-0.9"
        eng.addpath(eng.genpath(qetlab_path), nargout=0)

        # Test if PartialTrace function exists
        eng.eval("help PartialTrace", nargout=0)
        print("!!! QETLAB PartialTrace function found!")

        eng.quit()
        return True
    except Exception as e:
        print(f"XXX QETLAB test failed: {e}")
        return False


if __name__ == "__main__":
    # First test QETLAB installation
    if not test_qetlab_installation():
        print("Please install QETLAB first!")
        exit(1)

    # Test with a simple example: 2-qubit state |00><00|
    rho_full = np.zeros((4, 4))
    rho_full[0, 0] = 1.0  # |00><00|
    dims = [2, 2]
    subsys = [2]  # Trace out subsystem 2 (QETLAB uses 1-based indexing)

    try:
        reduced = matlab_partial_trace(rho_full, dims, subsys)
        print("Reduced density matrix (NumPy):")
        print(reduced)
        print("Expected: [[1, 0], [0, 0]] for |0><0| state")
    except Exception as e:
        print(f"Error: {e}")
