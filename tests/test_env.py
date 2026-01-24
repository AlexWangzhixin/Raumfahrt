print("Testing Python environment...")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test basic imports
try:
    import numpy
    print("NumPy imported successfully")
except Exception as e:
    print(f"NumPy import failed: {e}")

try:
    import matplotlib
    print("Matplotlib imported successfully")
except Exception as e:
    print(f"Matplotlib import failed: {e}")

try:
    import torch
    print("PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"PyTorch import failed: {e}")

print("Test completed!")
