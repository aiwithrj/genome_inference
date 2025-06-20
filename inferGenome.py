import argparse
import os
from parser import read_sham_file
from inference import compute_posteriors
from utils import write_output

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Genome Inference using Bayesian or Neural Network approach")

    parser.add_argument('--sham', required=True, help='Input .sham file containing reads')
    parser.add_argument('--out', type=str, help='Optional: Output file (.tsv). Defaults to output.tsv if not specified.')
    parser.add_argument('--p01', type=float, help='Error rate for 0 → 1 (Bayesian only)')
    parser.add_argument('--p10', type=float, help='Error rate for 1 → 0 (Bayesian only)')
    parser.add_argument('--prior1', type=float, help='Prior probability that base is 1 (Bayesian only)')
    parser.add_argument('--use-nn', action='store_true', help='Use Neural Network inference instead of Bayesian')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to saved neural network model')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration for Bayesian (via CuPy)')
    return parser.parse_args()

def pad_read(read, length):
    """
    Pads or trims the read to a fixed length.
    """
    return read[:length].ljust(length, '0')

def main():
    args = parse_args()

    # Load input .sham reads
    observations, max_pos = read_sham_file(args.sham)
    print(f"[DEBUG] Loaded {len(observations)} reads. Sample: {observations[0]}")

    # If output file not specified, default to "output.tsv"
    if not args.out:
        args.out = "output.tsv"

    # ----------------------------------------
    # Neural Network inference mode
    # ----------------------------------------
    if args.use_nn:
        import torch
        from nn_inference import nn_infer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using Neural Network inference on {device.upper()}")

        # Load model checkpoint and extract input_dim
        # We use map_location='cpu' to ensure compatibility, regardless of where the model was saved
        checkpoint = torch.load(args.model, map_location='cpu')
        input_dim = checkpoint["input_dim"]
        print(f"[INFO] Loaded model from {args.model} with input_dim = {input_dim}")

        # Pad all reads to uniform input length required by the model
        padded_observations = [(pos, pad_read(read, input_dim)) for pos, read in observations]

        # Run NN inference
        results = nn_infer(padded_observations, max_pos, args.model, device, input_dim)

    # ----------------------------------------
    # Bayesian inference mode
    # ----------------------------------------
    else:
        # Validate Bayesian parameters
        if args.p01 is None or args.p10 is None or args.prior1 is None:
            raise ValueError("Bayesian mode requires --p01, --p10, and --prior1 parameters.")

        print("[INFO] Using Bayesian inference " + ("with GPU (CuPy)" if args.use_gpu else "on CPU (NumPy)"))

        # Run Bayesian inference
        results = compute_posteriors(observations, max_pos, args.p01, args.p10, args.prior1, args.use_gpu)

    # Save output
    write_output(results, args.out)
    print(f"[DONE] Output written to: {args.out}")

if __name__ == '__main__':
    main()
