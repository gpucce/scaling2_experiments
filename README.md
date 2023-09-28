# scaling2.0

To run multiple experiments you have to create an `experiments.yaml` file with a list of `datasets` and `models` see the [cfg_templates.py](./src/cfg_templates.py) file to know the required fields.

Once that is in place, one should run
```
python -m src.generate_scripts --cfg experiments.yaml
```
Let us assume that in `experiments.yaml` one has set the following
```
...
sbatch_config:
  experiments_list_file_path: "exps_list.txt"
  sbatch_script_file_path: "sbatch_script.sbatch"
  ...
```

this will create to files:
 - The experiments_list file `exps_list.txt` holds
all the commands needed to run the experiments;
 - A sbatch script file `sbatch_script.sbatch`, that will start a slurm job array with as many experiments as there are pairs of datasets/models in `experiments.yaml` file.

Now one should be able to start all the jobs with the command:
```
python -m src.autorestart_job_array \
    "sbatch sbatch_script.sbatch" \
    --check-interval-secs 3 \ # change depending on the job
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --termination-str MADEITTOTHEEND \
    --verbose 1
```

make sure that `--output-file-template <this_path>` matches `sbatch_config.output`.


