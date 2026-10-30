#!/bin/bash -l
# Submit from the SIMCALC project directory:
# sbatch simcalc/run_solution.slurm abeta42_7q4m_8bfa_replica_2

# Slurm settings: one GPU, four CPU cores and 16 GB RAM on nvdfen3.
# Keep these directives above all executable commands.
#SBATCH --job-name=solution-md
#SBATCH --partition=gpu
#SBATCH --nodelist=nvdfen3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# Maximum elapsed time: seven days. %j below becomes the job number.
#SBATCH --time=7-00:00:00
#SBATCH --output=solution-md-%j.log

# Stop on unhandled command failures, including failures inside pipelines.
set -eo pipefail

# Read the first argument and require a single, safe directory name.
system_id=${1:-}
if [[ $# -ne 1 || -z "$system_id" || "$system_id" == . || "$system_id" == .. || "$system_id" == *[!a-zA-Z0-9_.-]* ]]; then
    echo "Usage: sbatch simcalc/run_solution.slurm SYSTEM_ID" >&2
    exit 1
fi

# Resolve all relative paths from the remote project directory.
cd /nfs/carlos/on-boarding-phospholipids-openmm

# Use separate inputs and outputs for the selected replica.
input_dir="data/inputs/openmm_native/solution/$system_id"
output_root="data/outputs/openmm_native/solution"
output_dir="$output_root/$system_id"
lock_dir="$output_root/.$system_id.lock"

# Check that both input directories exist before starting.
if [[ ! -d "$input_dir/openmm" || ! -d "$input_dir/toppar" ]]; then
    echo "Missing input directories for $system_id: $input_dir" >&2
    exit 1
fi

# Create missing parent folders; use a lock to prevent overlapping jobs.
mkdir -p "$output_root"
if ! mkdir "$lock_dir"; then
    echo "Another job may be running for $system_id (lock: $lock_dir)." >&2
    exit 1
fi

# Remove the lock on exit. A forced kill can leave it behind;
# check that no job is running before removing a leftover lock manually.
trap 'rmdir "$lock_dir"' EXIT

# Fresh runs only: protect existing results and handle resumption separately.
if [[ -e "$output_dir" ]]; then
    echo "Output already exists for $system_id: $output_dir" >&2
    echo "Stopping to protect existing results; check the job before resuming." >&2
    exit 1
fi

# Load CUDA and require GPU execution instead of falling back to the CPU.
module load cuda-12.9
export OPENMM_PLATFORM=CUDA

# Run equilibration and production. -u keeps log messages from being buffered.
.venv/bin/python -u mains/run_solution_sweep.py "$system_id"
