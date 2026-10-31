# Shared checks used before submission and again on the allocated node.
simcalc_error() {
    echo "ERROR: $*" >&2
    exit 1
}

simcalc_check_name() {
    [[ $# -eq 1 && -n "$1" && "$1" != . && "$1" != .. && "$1" != *[!a-zA-Z0-9_.-]* ]] ||
        simcalc_error "Provide one system ID containing only letters, digits, _, . or -."
}

simcalc_paths() {
    system_id=$1
    input_dir="$project_root/data/inputs/openmm_native/solution/$system_id/openmm"
    output_root="$project_root/data/outputs/openmm_native/solution"
    output_dir="$output_root/$system_id"
    execution_lock="$output_root/.$system_id.lock"
    log_root="$project_root/simcalc/logs"
    job_name="solution-md-$system_id"
}

simcalc_check_files() {
    local path name
    for path in "$project_root/pyproject.toml" "$project_root/uv.lock" \
        "$project_root/mains/run_solution_sweep.py" \
        "$project_root/simcalc/validate_solution.py" \
        "$project_root/simcalc/run_solution.slurm"; do
        [[ -f "$path" && -r "$path" ]] || simcalc_error "Missing or unreadable project file: $path. Upload the project first."
    done
    [[ -x "$project_root/.venv/bin/python" ]] ||
        simcalc_error "Missing executable .venv/bin/python. Install the NVIDIA environment on a compatible GPU node."
    for name in step3_input.psf step3_input.pdb step3_input.crd \
        step4_equilibration.inp step5_production.inp sysinfo.dat toppar.str; do
        path="$input_dir/$name"
        [[ -f "$path" && -r "$path" && -s "$path" ]] ||
            simcalc_error "Missing, empty or unreadable input: $path. Transfer the complete input bundle."
    done
}

simcalc_check_output() {
    [[ ! -e "$output_dir" && ! -L "$output_dir" ]] ||
        simcalc_error "Output already exists: $output_dir. Inspect it before choosing a new replica ID or a recovery procedure."
}

simcalc_check_fresh() {
    simcalc_check_output
    [[ ! -e "$execution_lock" && ! -L "$execution_lock" ]] ||
        simcalc_error "Execution lock exists: $execution_lock. Check squeue before considering manual cleanup."
}

simcalc_check_writable() {
    local directory probe
    for directory in "$output_root" "$log_root"; do
        mkdir -p -- "$directory" || simcalc_error "Cannot create directory: $directory. Check permissions and quota."
        probe=$(mktemp "$directory/.write-check.XXXXXX") ||
            simcalc_error "Cannot write to $directory. Check permissions, filesystem space and quota."
        if ! printf 'SIMCALC write check\n' > "$probe"; then
            rm -- "$probe"
            simcalc_error "Cannot write data to $directory. Check filesystem space and quota."
        fi
        rm -- "$probe" || simcalc_error "Cannot remove write-check file: $probe."
    done
}

# Slurm reports RAM in MiB and limits as days-hours:minutes:seconds.
simcalc_memory_mb() {
    local value=$1
    if [[ $value =~ ^([0-9]+)([MG]?)$ ]]; then
        case "${BASH_REMATCH[2]}" in
            G) echo "$(( ${BASH_REMATCH[1]} * 1024 ))" ;;
            *) echo "${BASH_REMATCH[1]}" ;;
        esac
    else
        simcalc_error "Unsupported memory setting: $value. Use MiB or a whole number followed by M or G."
    fi
}

simcalc_time_seconds() {
    local value=$1 days=0 hours=0 minutes=0 seconds=0
    case "$value" in UNLIMITED|INFINITE) echo 2147483647; return ;; esac
    if [[ $value == *-* ]]; then
        days=${value%%-*}
        value=${value#*-}
        [[ $days =~ ^[0-9]+$ ]] || simcalc_error "Invalid Slurm time: $1."
    fi
    if [[ $value =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        hours=${BASH_REMATCH[1]}; minutes=${BASH_REMATCH[2]}; seconds=${BASH_REMATCH[3]}
    elif [[ $value =~ ^([0-9]+):([0-9]+)$ ]]; then
        minutes=${BASH_REMATCH[1]}; seconds=${BASH_REMATCH[2]}
    elif [[ $value =~ ^[0-9]+$ ]]; then
        if (( days > 0 )); then hours=$value; else minutes=$value; fi
    else
        simcalc_error "Unsupported Slurm time: $1."
    fi
    echo "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}
