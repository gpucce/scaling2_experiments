python ./autorestart_job_array.py "sbatch sbatch_scripts/experiment_sbatch_script.sbatch" \
    --check-interval-secs 3 \
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --termination-str MADEITTOTHEEND \
    --verbose 1
