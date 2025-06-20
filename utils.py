import math

def log_prob(observed, genome_base, p01, p10):
    """
    Compute the log-probability of an observed base given the true genome base and error rates.

    Parameters:
        observed (str): Observed base in read ('0' or '1')
        genome_base (int): Actual genome base (0 or 1)
        p01 (float): Error rate for flipping from 0 → 1
        p10 (float): Error rate for flipping from 1 → 0

    Returns:
        float: Log-probability of the observation
    """
    if genome_base == 1:
        return math.log(1 - p10) if observed == '1' else math.log(p10)
    else:
        return math.log(p01) if observed == '1' else math.log(1 - p01)

def log_sum_exp(a, b):
    """
    Numerically stable computation of log(exp(a) + exp(b)).

    Parameters:
        a (float), b (float): Log-space values

    Returns:
        float: log(exp(a) + exp(b))
    """
    max_log = max(a, b)
    return max_log + math.log(math.exp(a - max_log) + math.exp(b - max_log))

def write_output(probabilities, output_file):
    """
    Write genome probabilities to a .tsv file.

    Parameters:
        probabilities (list): List of float probabilities per genome position
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        for pos, prob in enumerate(probabilities):
            f.write(f"{pos}\t{prob:.3f}\n")

