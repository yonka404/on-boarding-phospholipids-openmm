#!/bin/bash -l
# Submit with: bash simcalc/main.sh SYSTEM_ID
# Slurm chooses a GPU node. These directives must precede executable commands.
#SBATCH --job-name=solution-md
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# Seven days of elapsed time; %j becomes the job number.
#SBATCH --time=7-00:00:00
#SBATCH --output=solution-md-%j.log

set -eo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && -n ${SLURM_SUBMIT_DIR:-} ]] || {
    echo "ERROR: Submit through simcalc/main.sh; this script needs a Slurm allocation." >&2
    exit 1
}
project_root=$SLURM_SUBMIT_DIR
cd -- "$project_root"
source "$project_root/simcalc/common.sh"
simcalc_check_name "$@"
simcalc_paths "$1"

# Files may have changed while queued. Check again before touching replica output.
simcalc_check_files
simcalc_check_fresh
simcalc_check_writable
mkdir -- "$execution_lock" || simcalc_error "Could not acquire execution lock: $execution_lock. Check for another job."
trap 'rmdir -- "$execution_lock" || echo "WARNING: Could not remove execution lock: $execution_lock" >&2' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
simcalc_check_output

# The pinned OpenMM Linux wheels require glibc 2.34 or newer.
libc=$(getconf GNU_LIBC_VERSION) || simcalc_error "Cannot determine node glibc version."
if [[ $libc =~ ^glibc\ ([0-9]+)\.([0-9]+) ]]; then
    (( BASH_REMATCH[1] > 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] >= 34) )) ||
        simcalc_error "Node has $libc; OpenMM requires glibc 2.34+. Ask SIMCALC support about this node."
else
    simcalc_error "Unsupported node C library: $libc."
fi

# Keep Slurm's GPU mask. Device 0 means its first visible, allocated GPU.
module load cuda-12.9 || simcalc_error "Cannot load cuda-12.9. Check the node's available modules."
export OPENMM_PLATFORM=CUDA
export OPENMM_DEVICE_INDEX=0
.venv/bin/python -u simcalc/validate_solution.py "$system_id" ||
    simcalc_error "Allocation validation failed; the simulation was not started."

# The workflow validates inputs and initializes real CUDA simulations itself.
.venv/bin/python -u mains/run_solution_sweep.py "$system_id"
