# Try importing CuPy and running a small computation on the GPU
try:
    import cupy as cp  # Import CuPy library for GPU computations

    # Get GPU device info
    device = cp.cuda.Device()
    print(f"[SUCCESS] CuPy is using GPU: {device.name} (ID: {device.id})")

    # Perform a simple GPU operation: matrix multiplication
    a = cp.ones((1000, 1000))  # Create a 1000x1000 matrix of ones on the GPU
    b = cp.dot(a, a)           # Perform matrix multiplication on GPU
    print("[INFO] Matrix multiplication on GPU succeeded.")

# If CuPy is not installed at all
except ImportError:
    print("[ERROR] CuPy is not installed. Install with: pip install cupy-cuda11x")

# If there is a problem with the CUDA runtime (like driver mismatch, GPU unavailable, etc.)
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"[ERROR] CUDA runtime error: {e}")

# Catch-all for any other unexpected issues
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
