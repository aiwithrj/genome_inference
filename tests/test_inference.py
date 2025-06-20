import pytest
import numpy as np
from inference import compute_posteriors

def test_compute_posteriors_basic():
    """Test basic functionality of compute_posteriors with simple observations."""
    # Simple test case with a single observation
    observations = [(0, "101")]
    max_pos = 2
    p01 = 0.1  # false positive rate
    p10 = 0.2  # false negative rate
    prior1 = 0.5
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # Check result shape
    assert len(result) == max_pos + 1
    
    # Check that probabilities are between 0 and 1
    assert np.all(result >= 0) and np.all(result <= 1)
    
    # For position 0, observed "1" should increase probability from prior
    assert result[0] > prior1
    # For position 1, observed "0" should decrease probability from prior
    assert result[1] < prior1
    # For position 2, observed "1" should increase probability from prior
    assert result[2] > prior1


def test_compute_posteriors_dict_format():
    """Test compute_posteriors with dictionary format observations."""
    # Dictionary format observations
    observations = {
        0: ["101", "111"],
        5: ["010"]
    }
    max_pos = 7
    p01 = 0.1
    p10 = 0.2
    prior1 = 0.5
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # Check result shape
    assert len(result) == max_pos + 1
    
    # Positions with observations should differ from prior
    assert result[0] != prior1
    assert result[1] != prior1
    assert result[2] != prior1
    assert result[5] != prior1
    assert result[6] != prior1
    assert result[7] != prior1
    
    # Positions without observations should equal prior
    assert result[3] == prior1
    assert result[4] == prior1


def test_compute_posteriors_conflicting_evidence():
    """Test compute_posteriors with conflicting observations at the same position."""
    # Observations with conflicting evidence
    observations = [(0, "1"), (0, "0")]
    max_pos = 0
    p01 = 0.1
    p10 = 0.1
    prior1 = 0.5
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # With equal error rates and conflicting evidence, should be close to prior
    assert abs(result[0] - prior1) < 0.1


def test_compute_posteriors_strong_evidence():
    """Test compute_posteriors with strong evidence (many observations)."""
    # Many observations of the same value
    observations = [(0, "1")] * 10  # 10 observations of "1" at position 0
    max_pos = 0
    p01 = 0.1
    p10 = 0.1
    prior1 = 0.5
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # With many consistent observations, probability should be very high
    assert result[0] > 0.99


def test_compute_posteriors_error_rates():
    """Test compute_posteriors with different error rates."""
    observations = [(0, "1")]
    max_pos = 0
    
    # Low error rates
    result_low = compute_posteriors(observations, max_pos, 0.01, 0.01, 0.5)
    
    # High error rates
    result_high = compute_posteriors(observations, max_pos, 0.4, 0.4, 0.5)
    
    # With lower error rates, we should be more confident in our observation
    assert result_low[0] > result_high[0]


def test_compute_posteriors_prior_influence():
    """Test how different priors influence the posterior probabilities."""
    observations = [(0, "1")]
    max_pos = 0
    p01 = 0.1
    p10 = 0.1
    
    # Low prior
    result_low_prior = compute_posteriors(observations, max_pos, p01, p10, 0.1)
    
    # High prior
    result_high_prior = compute_posteriors(observations, max_pos, p01, p10, 0.9)
    
    # Higher prior should lead to higher posterior
    assert result_high_prior[0] > result_low_prior[0]


@pytest.mark.parametrize("use_gpu", [False, True])
def test_compute_posteriors_gpu_option(use_gpu):
    """Test that the GPU option works correctly."""
    observations = [(0, "1")]
    max_pos = 0
    p01 = 0.1
    p10 = 0.1
    prior1 = 0.5
    
    # This should work regardless of whether CuPy is available
    result = compute_posteriors(observations, max_pos, p01, p10, prior1, use_gpu=use_gpu)
    
    # Basic sanity check
    assert len(result) == max_pos + 1
    assert 0 <= result[0] <= 1


def test_compute_posteriors_empty_observations():
    """Test compute_posteriors with no observations."""
    observations = []
    max_pos = 5
    p01 = 0.1
    p10 = 0.1
    prior1 = 0.3
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # All positions should have the prior probability
    assert np.all(result == prior1)


def test_compute_posteriors_beyond_max_pos():
    """Test that observations beyond max_pos are ignored."""
    # Observation extends beyond max_pos
    observations = [(0, "11111")]
    max_pos = 2  # Only positions 0, 1, 2 are considered
    p01 = 0.1
    p10 = 0.1
    prior1 = 0.5
    
    # Run the function
    result = compute_posteriors(observations, max_pos, p01, p10, prior1)
    
    # Check result shape
    assert len(result) == max_pos + 1
    
    # Positions within max_pos should be updated
    assert result[0] != prior1
    assert result[1] != prior1
    assert result[2] != prior1
