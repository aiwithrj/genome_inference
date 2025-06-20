import pandas as pd
import matplotlib.pyplot as plt

def compare_outputs(cpu_file, gpu_file):
    # Load both output files
    cpu_df = pd.read_csv(cpu_file, sep="\t", header=None, names=["Position", "CPU_Bayes"])
    gpu_df = pd.read_csv(gpu_file, sep="\t", header=None, names=["Position", "GPU_Bayes"])

    # Merge on position
    merged = pd.merge(cpu_df, gpu_df, on="Position")

    # Calculate absolute difference
    merged["AbsDiff"] = (merged["CPU_Bayes"] - merged["GPU_Bayes"]).abs()

    # Print stats
    print("\n=== CPU vs GPU Bayesian Inference Comparison ===")
    print(f"Total Positions: {len(merged)}")
    print(f"Average Absolute Difference: {merged['AbsDiff'].mean():.6f}")
    print(f"Max Absolute Difference: {merged['AbsDiff'].max():.6f}")
    print(f"Min Absolute Difference: {merged['AbsDiff'].min():.6f}")
    print("===============================================")

    # Show first 10 rows
    print("\nSample Comparison:")
    print(merged.head(10))

    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(merged["Position"], merged["CPU_Bayes"], label="CPU (NumPy)", alpha=0.7)
    plt.plot(merged["Position"], merged["GPU_Bayes"], label="GPU (CuPy)", alpha=0.7)
    plt.title("CPU vs GPU Bayesian Inference Comparison")
    plt.xlabel("Position")
    plt.ylabel("Posterior Probability of 1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU Bayesian Inference Outputs")
    parser.add_argument("--cpu", required=True, help="Path to CPU (NumPy) output .tsv")
    parser.add_argument("--gpu", required=True, help="Path to GPU (CuPy) output .tsv")
    args = parser.parse_args()

    compare_outputs(args.cpu, args.gpu)