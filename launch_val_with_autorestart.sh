python -m src.autorestart_job_array "sbatch sbatch_scripts/val_experiment_sbatch_script.sbatch" \
    --check-interval-secs 5 \
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --verbose 1
