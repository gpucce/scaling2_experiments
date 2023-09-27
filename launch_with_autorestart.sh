python ./autorestart_job_array.py "sbatch test_slurm_arrays.sbatch" \
    --check-interval-secs 3 \
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --termination-str EXP \
    --verbose 1
