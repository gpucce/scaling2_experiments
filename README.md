# Slurm Experiments Runner

Create a vitrualenv and install the tiny requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run multiple experiments you have to create an `experiments.yaml` file with a list of `datasets` and `models` see the [cfg_templates.py](./src/cfg_templates.py) file to know the required fields. One can check [this train config](./configs/experiments.yaml) for a training example and [this val config](./configs/val_experiments.yaml) for a validation one.

Once that is in place, one should run

```bash
python -m src.generate_scripts --cfg val_experiments.yaml
```

Let us assume that in `experiments.yaml` one has set the following
```yaml
...
sbatch_config:
  experiments_list_file_path: "exps_list.txt"
  sbatch_script_file_path: "sbatch_script.sbatch"
  ...
```

this will create two files:
 - An experiments list file `exps_list.txt` which holds
all the commands needed to run the experiments;
 - A sbatch script file `sbatch_script.sbatch`, that will start a slurm job array with as many experiments as there are pairs of datasets/models in `experiments.yaml` file.

Now one should be able to start all the jobs with the command:

```python
sbatch sbatch_script.sbatch # autorestart coming soon
```

> NOTE: the logs paths are created when generating the sbatch scripts and not when running the actual experiments. Don't delete them.

## Autorestart [Experimental]

One can try using autorestart like this:

```bash
python -m src.autorestart_job_array \
    "sbatch sbatch_script.sbatch" \
    --check-interval-secs 3 \ # change depending on the job
    --output-file-template "slurm_logs/slurm-{job_id}_{array_task_id}.out" \
    --verbose 1
```
> NOTE: make sure that `--output-file-template <this_path>` matches `sbatch_config.output: <this_path>` . To do this one needs to change `{job_id} -> %A and {array_task_id} -> %a` so for using
> ```bash
> --output-file-template slurm-{job_id}_{array_task_id}.log
> ```
> in the config one will need
> ```yaml
> sbatch_config:
>   output: slurm-%A_%a.log
> ```


## TODO
- [ ] Add option to ignore experiments
- [ ] More testing
- [ ] Improve resume