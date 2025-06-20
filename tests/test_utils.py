import pytest
import math
import os
import tempfile
from utils import log_prob, log_sum_exp, write_output


class TestLogProb:
    """Tests for the log_prob function."""
    
    def test_correct_observations(self):
        """Test log_prob with correct observations (no errors)."""
        # When genome is 1 and we observe 1 (correct)
        log_p_correct_1 = log_prob('1', 1, 0.1, 0.1)
        # Should be log(1 - p10) = log(0.9)
        assert math.isclose(log_p_correct_1, math.log(0.9))
        
        # When genome is 0 and we observe 0 (correct)
        log_p_correct_0 = log_prob('0', 0, 0.1, 0.1)
        # Should be log(1 - p01) = log(0.9)
        assert math.isclose(log_p_correct_0, math.log(0.9))
    
    def test_error_observations(self):
        """Test log_prob with error observations."""
        # When genome is 1 but we observe 0 (error)
        log_p_error_1 = log_prob('0', 1, 0.1, 0.1)
        # Should be log(p10) = log(0.1)
        assert math.isclose(log_p_error_1, math.log(0.1))
        
        # When genome is 0 but we observe 1 (error)
        log_p_error_0 = log_prob('1', 0, 0.1, 0.1)
        # Should be log(p01) = log(0.1)
        assert math.isclose(log_p_error_0, math.log(0.1))
    
    @pytest.mark.parametrize("observed,genome,p01,p10,expected", [
        # Various combinations of observations, genome values, and error rates
        ('1', 1, 0.001, 0.001, math.log(0.999)),  # Very low error rate
        ('0', 0, 0.001, 0.001, math.log(0.999)),  # Very low error rate
        ('1', 1, 0.499, 0.499, math.log(0.501)),  # Very high error rate
        ('0', 0, 0.499, 0.499, math.log(0.501)),  # Very high error rate
        ('1', 1, 0.1, 0.3, math.log(0.7)),        # Different error rates
        ('0', 0, 0.1, 0.3, math.log(0.9))         # Different error rates
    ])
    def test_various_error_rates(self, observed, genome, p01, p10, expected):
        """Test log_prob with various error rates."""
        result = log_prob(observed, genome, p01, p10)
        assert math.isclose(result, expected)


class TestLogSumExp:
    """Tests for the log_sum_exp function."""
    
    def test_equal_values(self):
        """Test log_sum_exp with equal values."""
        # log(exp(0) + exp(0)) = log(2) ≈ 0.693
        result = log_sum_exp(0, 0)
        assert math.isclose(result, math.log(2))
    
    def test_very_different_values(self):
        """Test log_sum_exp with very different values."""
        # When one value is much larger, result should be close to the larger value
        result = log_sum_exp(100, 0)
        assert math.isclose(result, 100)
    
    def test_negative_values(self):
        """Test log_sum_exp with negative values."""
        # log(exp(-10) + exp(-20)) ≈ -10
        result = log_sum_exp(-10, -20)
        assert math.isclose(result, -10 + math.log(1 + math.exp(-10)))
    
    @pytest.mark.parametrize("a,b,expected", [
        (0.0, 0.0, math.log(2)),                  # Equal values
        (1.0, 2.0, math.log(math.exp(1) + math.exp(2))),  # Different values
        (-1.0, -2.0, math.log(math.exp(-1) + math.exp(-2))),  # Negative values
        (10.0, -10.0, math.log(math.exp(10) + math.exp(-10)))  # Very different values
    ])
    def test_various_combinations(self, a, b, expected):
        """Test log_sum_exp with various combinations of inputs."""
        result = log_sum_exp(a, b)
        assert math.isclose(result, expected)


class TestWriteOutput:
    """Tests for the write_output function."""
    
    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary file for testing output."""
        fd, filepath = tempfile.mkstemp(suffix='.tsv')
        os.close(fd)
        
        yield filepath
        
        # Clean up
        os.unlink(filepath)
    
    def test_basic_output(self, temp_output_file):
        """Test basic functionality of write_output."""
        # Sample probabilities
        probs = [0.1, 0.5, 0.9]
        
        # Write to file
        write_output(probs, temp_output_file)
        
        # Read back and check
        with open(temp_output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        assert lines[0] == "0\t0.100\n"
        assert lines[1] == "1\t0.500\n"
        assert lines[2] == "2\t0.900\n"
    
    def test_empty_list(self, temp_output_file):
        """Test write_output with an empty list."""
        # Empty probabilities list
        probs = []
        
        # Write to file
        write_output(probs, temp_output_file)
        
        # Read back and check
        with open(temp_output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 0
    
    def test_decimal_formatting(self, temp_output_file):
        """Test that probabilities are formatted with 3 decimal places."""
        # Probabilities with more than 3 decimal places
        probs = [0.12345, 0.67890]
        
        # Write to file
        write_output(probs, temp_output_file)
        
        # Read back and check
        with open(temp_output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        assert lines[0] == "0\t0.123\n"  # Truncated to 3 decimal places
        assert lines[1] == "1\t0.679\n"  # Truncated to 3 decimal places
    
    def test_large_dataset(self, temp_output_file):
        """Test write_output with a large dataset."""
        # Generate a large list of probabilities
        probs = [i / 1000 for i in range(1000)]
        
        # Write to file
        write_output(probs, temp_output_file)
        
        # Read back and check
        with open(temp_output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1000
        
        # Check a few random lines
        assert lines[0] == "0\t0.000\n"
        assert lines[500] == "500\t0.500\n"
        assert lines[999] == "999\t0.999\n"
