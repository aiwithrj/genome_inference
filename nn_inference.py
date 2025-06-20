import torch
import numpy as np
from model import GenomeMLP

def load_model(model_path):
    """
    Load the saved PyTorch GenomeMLP model and its input dimension.

    Args:
        model_path (str): Path to the saved model (.pth).

    Returns:
        model (torch.nn.Module): Loaded GenomeMLP model in eval mode.
        input_dim (int): Expected input length for the model.
    """

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    input_dim = checkpoint['input_dim']
    model = GenomeMLP(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, input_dim

def pad_read(read, target_len):
    """
    Pads or trims the read string to the target length.
    """
    return read[:target_len].ljust(target_len, '0')

def nn_infer(observations, max_pos, model_path,device, input_dim):
    """
    Perform neural network-based inference for genome positions.

    Args:
        observations (list): List of (start_pos, read_string) tuples.
        max_pos (int): Maximum genome position observed.
        model_path (str): Path to saved model checkpoint (.pth).
        device (str): 'cuda' or 'cpu'.
        input_dim (int): Input dimension expected by the model.

    Returns:
        np.ndarray: Posterior probabilities across genome positions.
    """
    model, loaded_dim = load_model(model_path)
    assert loaded_dim == input_dim, f"Mismatch in input_dim: model expects {loaded_dim}, got {input_dim}"
    model.to(device)
    model.eval()

    result_probs = np.zeros(max_pos + 1)
    counts = np.zeros(max_pos + 1)

    for start_pos, read in observations:
        padded = pad_read(read, input_dim)
        tensor_input = torch.tensor([[int(c) for c in padded]], dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor_input).numpy()[0]

        for i in range(len(read)):
            genome_pos = start_pos + i
            if genome_pos <= max_pos:
                result_probs[genome_pos] += output[i]
                counts[genome_pos] += 1

    for i in range(len(result_probs)):
        if counts[i] > 0:
            result_probs[i] /= counts[i]
        else:
            result_probs[i] = 0.5  # Default prior for uncovered positions

    return result_probs