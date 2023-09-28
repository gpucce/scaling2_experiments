python -m src.autorestart_job_array "sbatch --array=1-6 sbatch_scripts/experiment_sbatch_script.sbatch" \
    --check-interval-secs 5 \
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --termination-str MADEITTOTHEEND \
    --verbose 1
