#!/bin/bash
# Run through PyCharm's Remote SSH External Tool, or on the SIMCALC login node:
# bash simcalc/main.sh [--check-only] SYSTEM_ID
set -eo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$project_root/simcalc/common.sh"
check_only=false
if [[ ${1:-} == --check-only ]]; then check_only=true; shift; fi
simcalc_check_name "$@"
simcalc_paths "$1"
cd -- "$project_root"

# These checks use files and Slurm only; the login node cannot import OpenMM.
simcalc_check_files
simcalc_check_fresh
simcalc_check_writable
for command in sbatch scontrol sinfo squeue df; do
    command -v "$command" >/dev/null || simcalc_error "Required command is missing: $command. Run this on the SIMCALC login node."
done

# Serialize submissions for this replica, then check for an existing queued job.
submission_lock="$log_root/.$system_id.submit.lock"
mkdir -- "$submission_lock" || simcalc_error "Submission lock exists: $submission_lock. Check for another launcher before manual cleanup."
trap 'rmdir -- "$submission_lock" || echo "WARNING: Could not remove submission lock: $submission_lock" >&2' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
controller=$(scontrol ping) || simcalc_error "Slurm controller is unreachable. Reconnect or contact SIMCALC support."
[[ $controller == *"is UP"* ]] || simcalc_error "Slurm controller is not UP: $controller"
queue=$(squeue -h -u "$(id -un)" -o '%i|%j|%T') || simcalc_error "Cannot query your Slurm jobs."
while IFS='|' read -r queued_id queued_name queued_state; do
    [[ $queued_name != "$job_name" ]] || simcalc_error "Job $queued_id is already $queued_state for $system_id. Do not submit it twice."
done <<< "$queue"

# Read the resource requests from the batch file so there is one source of truth.
batch_file="$project_root/simcalc/run_solution.slurm"
setting() {
    awk -v key="--$1=" '$1 == "#SBATCH" && index($2, key) == 1 {print substr($2, length(key) + 1)}' "$batch_file"
}
partition=$(setting partition)
cpus=$(setting cpus-per-task)
memory=$(simcalc_memory_mb "$(setting mem)")
walltime=$(simcalc_time_seconds "$(setting time)")
[[ -n $partition && $cpus =~ ^[1-9][0-9]*$ ]] || simcalc_error "Invalid partition or CPU request in $batch_file."
partition_info=$(scontrol show partition "$partition" -o) || simcalc_error "Cannot read partition $partition."
field() {
    awk -v key="$1=" '{for (i=1; i<=NF; i++) if (index($i,key)==1) print substr($i,length(key)+1)}' <<< "$partition_info"
}
[[ $(field State) == UP ]] || simcalc_error "Partition $partition is not UP."
max_time=$(field MaxTime)
[[ -n $max_time ]] || simcalc_error "Cannot determine partition $partition walltime limit."
(( walltime <= $(simcalc_time_seconds "$max_time") )) || simcalc_error "Requested walltime exceeds partition $partition limit ($max_time)."

# Check total capacity, not currently free capacity: busy nodes can queue work.
nodes=$(sinfo -N -h -p "$partition" -o '%N|%T|%G|%c|%m') || simcalc_error "Cannot read GPU node availability."
printf 'GPU node availability (node | state | GPUs | CPUs | RAM MiB):\n%s\n' "$nodes"
if ! awk -F '|' -v cpus="$cpus" -v memory="$memory" '
    { if(tolower($2) ~ /down|drain|fail|invalid|unknown|maint/) next;
      gpu=0; count=split($3,resources,",");
      for(i=1;i<=count;i++) {sub(/\(.*/,"",resources[i]); n=split(resources[i],parts,":"); if(parts[1]=="gpu") gpu+=parts[n]+0}
      if (gpu>=1 && $4+0>=cpus && $5+0>=memory) capable=1
    }
    END {exit !capable}' <<< "$nodes"; then
    simcalc_error "No usable node in $partition has capacity for one GPU, $cpus CPUs and $memory MiB RAM. Check node health and resource requests."
fi
printf '\nFilesystem space (this does not check your quota or predict trajectory size):\n'
df -h -- "$output_root" "$log_root" || simcalc_error "Cannot inspect filesystem space."

# Prediction is advisory; a missing estimate must not prevent normal queueing.
options=(--chdir="$project_root" --job-name="$job_name" --output="$log_root/$system_id-%j.log")
if ! sbatch --test-only "${options[@]}" "$batch_file" "$system_id"; then
    echo "WARNING: Start estimate unavailable. Actual submission will determine acceptance." >&2
fi
if $check_only; then
    echo "Checks passed. No simulation job was submitted."
    exit 0
fi
simcalc_check_fresh
if ! reply=$(sbatch --parsable "${options[@]}" "$batch_file" "$system_id"); then
    simcalc_error "Slurm rejected submission. Review its diagnostic above before retrying."
fi
job_id=${reply%%;*}
[[ $job_id =~ ^[0-9]+$ ]] || simcalc_error "Slurm replied: $reply. Submission may have succeeded; inspect squeue before retrying."
echo "Submitted system $system_id as job $job_id."
echo "Log: $log_root/$system_id-$job_id.log"
echo "Status: squeue -j $job_id"
echo "Follow log after the job starts: tail -f '$log_root/$system_id-$job_id.log'"
