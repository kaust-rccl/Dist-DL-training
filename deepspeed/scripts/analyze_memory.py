import pandas as pd
import argparse
import os
import glob

# ----------------------------------------------
# Analyze GPU Memory Logs
# ----------------------------------------------
def analyze_gpu(path):
    """
    Analyzes GPU memory usage from all .csv logs inside path/gpu_memory/.
    Each CSV may contain logs for one or multiple GPUs, with memory usage samples over time.

    Args:
        path (str): Absolute or relative path to the job-specific log directory (contains gpu_memory/).
    """
    gpu_log_dir = os.path.join(path, "gpu_memory")

    # Check if the GPU memory log directory exists
    if not os.path.isdir(gpu_log_dir):
        print(f"[!] GPU log directory not found: {gpu_log_dir}")
        return

    # Find all .csv GPU logs in the directory (one per node typically)
    log_files = glob.glob(os.path.join(gpu_log_dir, "*.csv"))
    if not log_files:
        print(f"[!] No GPU memory logs found in {gpu_log_dir}")
        return

    print("\nGPU Memory Usage")

    # Process each CSV file separately (1 per node or per GPU set)
    for file in sorted(log_files):
        label = os.path.basename(file).replace(".csv", "")  # e.g., gpu_memory_log_gpu214-10

        # NVIDIA-SMI logs have one header row we skip
        df = pd.read_csv(file, skiprows=1, header=None)
        df.columns = ["timestamp", "index", "name", "memory_used", "memory_total"]

        # Loop over all unique GPU indices (in case multiple GPUs are recorded in the same file)
        for idx in sorted(df["index"].unique()):
            gpu_df = df[df["index"] == idx]
            peak = gpu_df["memory_used"].max()
            avg = gpu_df["memory_used"].mean()
            print(f"[{label} - GPU {idx}] Peak = {peak:.0f} MiB, Avg = {avg:.2f} MiB")


# ----------------------------------------------
# Analyze CPU Memory Logs
# ----------------------------------------------
def analyze_cpu(path):
    """
    Analyzes CPU memory usage from a psrecord-formatted log at path/cpu_memory/cpu_memory_log.txt.

    Args:
        path (str): Absolute or relative path to the job-specific log directory (contains cpu_memory/).
    """
    cpu_log_dir = os.path.join(path, "cpu_memory")
    log_path = os.path.join(cpu_log_dir, "cpu_memory_log.txt")

    # Check if the log file exists
    if not os.path.exists(log_path):
        print(f"[!] CPU log not found: {log_path}")
        return

    # Read the CPU log, skipping comments, splitting by whitespace
    df = pd.read_csv(log_path, sep=r"\s+", comment="#", header=None)

    # psrecord outputs: Elapsed, CPU%, Real(MB), Virtual(MB) → Real = 3rd column (index 2)
    if df.shape[1] < 3:
        print(f"[!] Unexpected CPU log format in {log_path}")
        return

    real = df.iloc[:, 2]
    peak = real.max()
    avg = real.mean()

    print("\nCPU Memory Usage")
    print(f"   Peak:   {peak:.0f} MB")
    print(f"   Average:{avg:.2f} MB")


# ----------------------------------------------
# CLI Entrypoint
# ----------------------------------------------
def main():
    """
    Parses CLI arguments and triggers memory analysis for GPU and CPU logs.
    Expects:
        - job_id (str): Used to locate logs under path/{job_id}/
        - --path (str): Base path containing job ID directories with memory logs.
    """
    parser = argparse.ArgumentParser(description="Analyze memory logs for a given job")
    parser.add_argument("job_id", help="SLURM job ID used to locate log directory")
    parser.add_argument("--path", required=True, help="Base directory where job directories are stored")

    args = parser.parse_args()

    job_dir = os.path.join(args.path, args.job_id)
    print(f"Analyzing memory logs in: {job_dir}")

    analyze_gpu(job_dir)
    analyze_cpu(job_dir)


# Entry point: script only executes if run directly
if __name__ == "__main__":
    main()
