#!/bin/bash -l
# Submit with: python3 simcalc/main.py SYSTEM_ID
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
cd -- "${SLURM_SUBMIT_DIR:?Submit through simcalc/main.py SYSTEM_ID}"

# Modules change the shell environment inherited by Python, so this stays in Bash.
module load cuda-12.9 || {
    echo "ERROR: Cannot load cuda-12.9. Check the node's available modules." >&2
    exit 1
}

# Python owns all checks and locks. exec lets it receive Slurm's interruption signals.
exec /usr/bin/python3 -u simcalc/run_solution.py "$@"
