import torch
import argparse
from model import GenomeMLP
from parser import read_sham_file

def get_max_read_len(observations):
    """
    Computes the maximum read length from the .sham observations.

    Parameters:
        observations (list): List of (start_position, read_string) tuples.

    Returns:
        int: Maximum read length found in the dataset.
    """
    return max(len(read) for _, read in observations)

def save_model(model_path, input_dim):
    """
    Initializes and saves a dummy neural network model with specified input dimensions.

    Parameters:
        model_path (str): Path to save the model.
        input_dim (int): Input dimension for the neural network (i.e., max read length).
    """
    model = GenomeMLP(input_dim)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim
    }, model_path)
    print(f"[INFO] Dummy model saved at: {model_path} with input_dim = {input_dim}")

def main():
    parser = argparse.ArgumentParser(description="Generate and save a dummy NN model based on .sham file structure")
    parser.add_argument('--sham', required=True, help='Path to .sham file for detecting input dimension')
    parser.add_argument('--output', type=str, default='mymodel.pth', help='Output path for saved model')
    args = parser.parse_args()

    # Read sham data and calculate input dimension
    observations, _ = read_sham_file(args.sham)
    max_read_len = get_max_read_len(observations)

    # Save dummy model
    save_model(args.output, max_read_len)

if __name__ == "__main__":
    main()

