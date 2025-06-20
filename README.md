# Genome Inference Tool

This project implements **Bayesian** and **Neural Network** based genome inference on `.sham` formatted sequencing reads.

##  Directory Structure

```
genome-inference/
├── compare_outputs.py       # Script to compare CPU vs GPU inference results
├── example.sham             # Small example input file
├── generate_large_sham.py   # Script to generate synthetic test data
├── gpu_test.py              # Script to test GPU/CuPy functionality
├── inference.py             # Core Bayesian inference implementation
├── inferGenome.py           # Main CLI entry point
├── large_example.sham       # Larger synthetic test dataset
├── model.py                 # Neural network model definition
├── mymodel.pth              # Saved neural network model (small)
├── mymodel_large.pth        # Saved neural network model (large)
├── nn_inference.py          # Neural network inference implementation
├── output_*.tsv             # Various output files
├── parser.py                # Input file parser
├── requirements.txt         # Project dependencies
├── save_dummy_model.py      # Script to create and save a neural network model
├── utils.py                 # Utility functions
└── tests/                   # Test directory
    ├── __init__.py
    ├── test_inference.py    # Unit tests for Bayesian inference
    ├── test_nn_inference.py # Unit tests for Neural Network inference
    ├── test_parser.py       # Unit tests for parser functionality
    ├── test_utils.py        # Unit tests for utility functions
    └── test_inferGenome.py  # Unit tests for CLI interface
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies in `requirements.txt`:
```
memory-profiler
torch
numpy
cupy-cuda11x    # For GPU acceleration
```

##  Input Format: `.sham` File

Each line in a `.sham` file should be:

```
<start_position>\t<binary_read_string>
```

Example:
```
1	1001110101000
4	1010111001111
13	1101011
```

##  Usage

### 1. Run Bayesian Inference (CPU)

```bash
python inferGenome.py --sham example.sham --out output_bayes.tsv --p01 0.01 --p10 0.01 --prior1 0.5
```

### 2. Run Bayesian Inference (GPU)

```bash
python inferGenome.py --sham example.sham --out output_bayes_gpu.tsv --p01 0.01 --p10 0.01 --prior1 0.5 --use-gpu
```

### 3. Run Bayesian Inference on Large Dataset (GPU)

```bash
python inferGenome.py --sham large_example.sham --out output_bayes_large_gpu.tsv --p01 0.01 --p10 0.01 --prior1 0.5 --use-gpu
```

### 4. Run Neural Network Inference

```bash
python inferGenome.py --sham example.sham --out output_nn.tsv --use-nn --model mymodel.pth
```

### 5. Run Neural Network Inference on Large Dataset

```bash
python inferGenome.py --sham large_example.sham --out output_nn_large.tsv --use-nn --model mymodel_large.pth
```

### 6. Compare CPU vs GPU Results

```bash
python compare_outputs.py --cpu output_bayes.tsv --gpu output_bayes_gpu.tsv
```

##  Neural Network Setup

To generate a dummy NN model:

```bash
python save_dummy_model.py --sham example.sham --output mymodel.pth
```

This initializes and saves a simple MLP model for testing. The model architecture is:
- Input layer (size based on max read length)
- Hidden layer with 64 neurons and ReLU activation
- Output layer (same size as input) with Sigmoid activation

##  Generate Synthetic Data

```bash
python generate_large_sham.py --num-reads 100000 --max-pos 1000 --max-len 50 --p1 0.5 --out large_example.sham
```

##  GPU Acceleration

### Testing GPU Availability

```bash
python gpu_test.py
```

This script tests if CuPy is properly installed and can access your GPU. If successful, you'll see:
```
[SUCCESS] CuPy is using GPU: <your-gpu-name> (ID: 0)
[INFO] Matrix multiplication on GPU succeeded.
```

If CuPy is not installed or GPU is not available, you'll see an error message.

### GPU Fallback Mechanism

The code includes an automatic fallback mechanism:

1. When you use the `--use-gpu` flag, the system first checks if CuPy is installed and a GPU is available.
2. If both conditions are met, it will use GPU acceleration and show:
   ```
   [INFO] Using Bayesian inference with GPU (CuPy)
   [DEBUG] compute_posteriors: using CuPy (GPU)
   ```
3. If CuPy is not installed or no GPU is available, it will automatically fall back to CPU:
   ```
   [INFO] Using Bayesian inference with GPU (CuPy)
   [DEBUG] compute_posteriors: using NumPy (CPU)
   ```

This design ensures the code works on both GPU and non-GPU systems without modification.

### Requirements for GPU Acceleration

To use GPU acceleration, you need:
1. A compatible NVIDIA GPU
2. CUDA drivers installed
3. CuPy package installed (`pip install cupy-cuda11x` or appropriate version)

### Bayesian Inference with GPU

Basic usage:
```bash
python inferGenome.py --sham example.sham --use-gpu --p01 0.01 --p10 0.01 --prior1 0.5 --out output_bayes_gpu.tsv
```

With large dataset:
```bash
python inferGenome.py --sham large_example.sham --use-gpu --p01 0.01 --p10 0.01 --prior1 0.5 --out output_bayes_large_gpu.tsv
```

For large datasets, ensure sufficient GPU memory is available.

### Neural Network GPU Acceleration

Neural networks automatically use GPU if available via PyTorch:

```bash
python inferGenome.py --sham example.sham --use-nn --model mymodel.pth --out output_nn_gpu.tsv
```

PyTorch has its own GPU detection mechanism and will show:
```
[INFO] Using Neural Network inference on CUDA
```
if a GPU is available, or:
```
[INFO] Using Neural Network inference on CPU
```
if no GPU is available.

##  Output Format

Output `.tsv` contains:
```
<position>\t<posterior_probability>
```

Example:
```
0	0.500
1	0.990
2	0.010
...
```

##  Testing

Run all unit tests with:

```bash
python -m unittest discover tests
```

Run specific test modules:

```bash
# Test Bayesian inference
python -m unittest tests.test_inference

# Test Neural Network inference
python -m unittest tests.test_nn_inference

# Test parser functionality
python -m unittest tests.test_parser

# Test utility functions
python -m unittest tests.test_utils

# Test CLI interface
python -m unittest tests.test_inferGenome
```

The test suite includes:

1. **Bayesian Inference Tests**:
   - Basic inference functionality
   - Edge cases (no observations, single observation)
   - Extreme error rates
   - Conflicting observations
   - Strong priors
   - GPU vs CPU result comparison

2. **Neural Network Tests**:
   - Model loading and saving
   - Read padding
   - Inference with various input types
   - Empty and overlapping observations

3. **Parser Tests**:
   - File format validation
   - Edge cases (empty files, malformed lines)
   - Position handling (negative, zero)

4. **Utility Function Tests**:
   - Log probability calculations
   - Numerically stable operations
   - Output formatting

5. **CLI Interface Tests**:
   - Command-line argument parsing
   - Integration of components

## 🔬 Notes for Real Sequencing Data

- Input must be `.sham` formatted
- Bayesian model is stable and interpretable
- NN model here is a **dummy** and should be retrained with actual labeled data
- For large genomes, consider using the GPU acceleration option
- Memory usage scales with genome size; monitor with `memory-profiler`

## 👤 Author

Rupesh Jonnalagadda — 2025

## 📘 License

MIT License
