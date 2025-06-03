import pandas as pd
import argparse
import os
import sys


# ------------------------------------------
# Analyze GPU Memory Log
# ------------------------------------------
def analyze_gpu(job_id):
    # Path to the GPU memory CSV log
    gpu_log_path = f"gpu_memory/gpu_memory_log_{job_id}.csv"

    # Check if the file exists
    if not os.path.exists(gpu_log_path):
        print(f"[!] GPU log not found: {gpu_log_path}")
        return

    # Load the CSV, skipping the first line which contains column headers from nvidia-smi
    df = pd.read_csv(gpu_log_path, skiprows=1, header=None)
    df.columns = ["timestamp", "index", "name", "memory_used", "memory_total"]  # Manually set column names

    # Calculate peak and average GPU memory usage
    peak = df["memory_used"].max()
    avg = df["memory_used"].mean()

    # Display results
    print(f"GPU Memory Usage (MiB)")
    print(f"   Peak:   {peak:.0f}")
    print(f"   Average:{avg:.2f}")


# ------------------------------------------
# Analyze CPU Memory Log
# ------------------------------------------
def analyze_cpu(job_id):
    # Path to the CPU memory log (psrecord output)
    cpu_log_path = f"cpu_memory/cpu_memory_log_{job_id}.txt"

    # Check if the file exists
    if not os.path.exists(cpu_log_path):
        print(f"[!] CPU log not found: {cpu_log_path}")
        return

    # Read the log, skipping any commented lines (e.g., header starting with #)
    df = pd.read_csv(cpu_log_path, comment="#", delim_whitespace=True)

    # Calculate peak and average real (resident) memory usage
    peak = df["Real"].max()
    avg = df["Real"].mean()

    # Display results
    print(f"\nCPU Memory Usage (MB)")
    print(f"   Peak:   {peak:.0f}")
    print(f"   Average:{avg:.2f}")


# ------------------------------------------
# CLI Entrypoint
# ------------------------------------------
def main():
    # Set up CLI argument parser
    parser = argparse.ArgumentParser(description="Analyze GPU and CPU memory logs by JOB_ID")
    parser.add_argument("job_id", help="SLURM job ID used in log filenames")

    # Parse arguments
    args = parser.parse_args()

    print(f"Analyzing logs for JOB_ID: {args.job_id}")

    # Run analysis for both GPU and CPU logs
    analyze_gpu(args.job_id)
    analyze_cpu(args.job_id)


# Run the script if executed directly
if __name__ == "__main__":
    main()
