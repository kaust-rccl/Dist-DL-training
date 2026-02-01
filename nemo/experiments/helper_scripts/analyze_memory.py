import pandas as pd
import argparse
import os
import glob


# ----------------------------------------------
# Analyze GPU Memory Logs
# ----------------------------------------------
def analyze_gpu(base_dir, job_id):
    """
    Analyzes GPU memory logs from gpu_memory/<job_id>/.

    Supports:
    - Single-GPU: one file, one GPU
    - Multi-GPU: one file, multiple GPUs
    - Multi-node: multiple files, each with one or more GPUs

    Args:
        base_dir (str): Base path containing the 'gpu_memory' directory
        job_id (str): Job ID used to locate logs inside 'gpu_memory/<job_id>/'
    """
    gpu_log_dir = os.path.join(base_dir, "gpu_memory", job_id)

    if not os.path.isdir(gpu_log_dir):
        print(f"[!] GPU log directory not found: {gpu_log_dir}")
        return

    log_files = glob.glob(os.path.join(gpu_log_dir, "*.csv"))
    if not log_files:
        print(f"[!] No GPU logs found in {gpu_log_dir}")
        return

    print("\nGPU Memory Usage")

    for filepath in sorted(log_files):
        filename = os.path.basename(filepath).replace(".csv", "")
        try:
            df = pd.read_csv(filepath, skiprows=1, header=None)
            df.columns = ["timestamp", "index", "name", "memory_used", "memory_total"]
        except Exception as e:
            print(f"[!] Failed to read {filepath}: {e}")
            continue

        for gpu_id in sorted(df["index"].unique()):
            gpu_df = df[df["index"] == gpu_id]
            peak = gpu_df["memory_used"].max()
            avg = gpu_df["memory_used"].mean()
            mode = gpu_df["memory_used"].mode()
            mode_str = ", ".join(f"{m:.0f}" for m in mode)
            print(f"[{filename} - GPU {gpu_id}] Peak = {peak:.0f} MiB, Avg = {avg:.2f} MiB, Mode = {mode_str} MiB")


# ----------------------------------------------
# Analyze CPU Memory Logs
# ----------------------------------------------
def analyze_cpu(base_dir, job_id):
    """
    Analyzes CPU memory logs from cpu_memory/<job_id>/cpu_memory_log.txt.

    Args:
        base_dir (str): Base path containing the 'cpu_memory' directory
        job_id (str): Job ID used to locate logs inside 'cpu_memory/<job_id>/'
    """
    log_path = os.path.join(base_dir, "cpu_memory", job_id, "cpu_memory_log.txt")

    if not os.path.exists(log_path):
        print(f"[!] CPU log not found: {log_path}")
        return

    try:
        df = pd.read_csv(log_path, sep=r"\s+", comment="#", header=None)
        if df.shape[1] < 3:
            raise ValueError("Log format has insufficient columns.")
        real = df.iloc[:, 2]
        peak = real.max()
        avg = real.mean()
        mode = real.mode()
        mode_str = ", ".join(f"{m:.0f}" for m in mode)

        print("\nCPU Memory Usage")
        print(f"   Peak =    {peak:.0f} MB")
        print(f"   Average = {avg:.2f} MB")
        print(f"   Mode =    {mode_str} MB")
    except Exception as e:
        print(f"[!] Failed to read {log_path}: {e}")


# ----------------------------------------------
# CLI Entrypoint
# ----------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze GPU and CPU memory logs for a job")
    parser.add_argument("job_id", help="SLURM Job ID (used as log folder name)")
    parser.add_argument("--path", default=".", help="Root directory (default: current directory)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu-only", action="store_true", help="Analyze only GPU logs")
    group.add_argument("--cpu-only", action="store_true", help="Analyze only CPU logs")

    args = parser.parse_args()

    print(f"Analyzing logs for job ID: {args.job_id} under {args.path}")

    if args.gpu_only:
        analyze_gpu(args.path, args.job_id)
    elif args.cpu_only:
        analyze_cpu(args.path, args.job_id)
    else:
        analyze_gpu(args.path, args.job_id)
        analyze_cpu(args.path, args.job_id)


if __name__ == "__main__":
    main()
