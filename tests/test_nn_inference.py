import pytest
import torch
import numpy as np
import os
import tempfile
from nn_inference import nn_infer, pad_read, load_model
from model import GenomeMLP


@pytest.fixture
def dummy_model():
    """Create a temporary dummy model for testing."""
    # Create a temporary file for the model
    fd, model_path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    
    # Model parameters
    input_dim = 10
    
    # Create and save a dummy model
    model = GenomeMLP(input_dim)
    
    # Initialize weights to a deterministic pattern for consistent test results
    # This makes the first position always predict close to 1.0, and others decrease linearly
    with torch.no_grad():
        # Set first layer weights to make first position important
        weights = torch.zeros(64, input_dim)
        for i in range(input_dim):
            weights[:, i] = 1.0 - (i * 0.1)  # Decreasing importance
        model.model[0].weight.copy_(weights)
        
        # Set biases to small positive values
        model.model[0].bias.fill_(0.1)
        
        # Set second layer to amplify the pattern
        model.model[2].weight.fill_(0.5)
        model.model[2].bias.fill_(0.1)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim
    }, model_path)
    
    # Return model info
    yield {
        'path': model_path,
        'input_dim': input_dim
    }
    
    # Clean up after tests
    os.unlink(model_path)


def test_pad_read_shorter():
    """Test padding a read that's shorter than the target length."""
    result = pad_read("101", 5)
    assert result == "10100"
    assert len(result) == 5


def test_pad_read_longer():
    """Test trimming a read that's longer than the target length."""
    result = pad_read("1010101", 5)
    assert result == "10101"
    assert len(result) == 5


def test_pad_read_exact():
    """Test a read that's already the target length."""
    result = pad_read("10101", 5)
    assert result == "10101"
    assert len(result) == 5


@pytest.mark.parametrize("read,length,expected", [
    ("", 3, "000"),           # Empty read
    ("1", 1, "1"),            # Single character, exact length
    ("0000", 10, "0000000000"),  # Pad with many zeros
    ("11111", 3, "111"),      # Significant trimming
    ("101010", 6, "101010"),  # Exact match
])
def test_pad_read_various(read, length, expected):
    """Test various padding/trimming scenarios."""
    result = pad_read(read, length)
    assert result == expected
    assert len(result) == length


def test_load_model(dummy_model):
    """Test that a model can be loaded correctly."""
    # Load the model
    model, input_dim = load_model(dummy_model['path'])
    
    # Check input dimension is preserved
    assert input_dim == dummy_model['input_dim']
    
    # Check model structure
    assert isinstance(model, GenomeMLP)
    
    # Check model is in eval mode (not training mode)
    assert not model.training


def test_nn_infer_empty_observations(dummy_model):
    """Test neural network inference with no observations."""
    # Empty observations list
    observations = []
    max_pos = 5
    
    # Run inference
    results = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results) == max_pos + 1
    
    # Check all positions have default prior (0.5)
    for prob in results:
        assert prob == 0.5


def test_nn_infer_single_observation(dummy_model):
    """Test neural network inference with a single observation."""
    # Single observation with all 1s
    observations = [(0, "1" * dummy_model['input_dim'])]
    max_pos = dummy_model['input_dim'] - 1
    
    # Run inference
    results = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results) == max_pos + 1
    
    # Check all probabilities are between 0 and 1
    for prob in results:
        assert 0 <= prob <= 1
    
    # Check that the model produces some output
    # The exact values will depend on the random initialization
    assert results[0] != 0.5  # Should be influenced by the observation


def test_nn_infer_multiple_observations(dummy_model):
    """Test neural network inference with multiple observations."""
    # Multiple observations at different positions
    observations = [
        (0, "1" * 5),
        (5, "1" * 5)
    ]
    max_pos = 10
    
    # Run inference
    results = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results) == max_pos + 1
    
    # Check all probabilities are between 0 and 1
    for prob in results:
        assert 0 <= prob <= 1
    
    # Positions with observations should differ from default prior
    assert results[0] != 0.5
    assert results[5] != 0.5
    
    # Positions without observations should have default prior
    assert results[10] == 0.5


def test_nn_infer_overlapping_observations(dummy_model):
    """Test neural network inference with overlapping reads."""
    # Overlapping observations
    observations = [
        (3, "111"),
        (5, "111"),
        (4, "111")
    ]
    max_pos = 10
    
    # Run inference
    results = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results) == max_pos + 1
    
    # Position 5 should have contributions from all 3 reads
    # and should differ from the default prior
    assert results[5] != 0.5
    
    # Positions without observations should have default prior
    assert results[0] == 0.5
    assert results[10] == 0.5


def test_nn_infer_input_dim_mismatch(dummy_model):
    """Test that an error is raised when input dimensions don't match."""
    observations = [(0, "1" * dummy_model['input_dim'])]
    max_pos = 5
    wrong_dim = dummy_model['input_dim'] + 1
    
    # Should raise an assertion error due to dimension mismatch
    with pytest.raises(AssertionError) as excinfo:
        nn_infer(observations, max_pos, dummy_model['path'], "cpu", wrong_dim)
    
    # Check error message mentions input_dim
    assert "input_dim" in str(excinfo.value)


def test_nn_infer_device_handling(dummy_model):
    """Test that device specification works correctly."""
    observations = [(0, "1" * dummy_model['input_dim'])]
    max_pos = 5
    
    # Run inference with explicit CPU device
    results_cpu = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results_cpu) == max_pos + 1
    
    # If CUDA is available, test with GPU device
    if torch.cuda.is_available():
        results_gpu = nn_infer(observations, max_pos, dummy_model['path'], "cuda", dummy_model['input_dim'])
        assert len(results_gpu) == max_pos + 1
        
        # Results should be similar between CPU and GPU
        assert np.allclose(results_cpu, results_gpu, atol=1e-5)


def test_nn_infer_long_reads(dummy_model):
    """Test inference with reads longer than the model's input dimension."""
    # Create a read longer than the input dimension
    long_read = "1" * (dummy_model['input_dim'] * 2)
    observations = [(0, long_read)]
    max_pos = dummy_model['input_dim'] - 1
    
    # Run inference
    results = nn_infer(observations, max_pos, dummy_model['path'], "cpu", dummy_model['input_dim'])
    
    # Check result length
    assert len(results) == max_pos + 1
    
    # The model should only use the first input_dim characters
    # All positions within max_pos should have predictions
    for i in range(max_pos + 1):
        assert 0 <= results[i] <= 1
