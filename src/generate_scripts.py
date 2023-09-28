from clize import run
import pandas as pd
import math
from pathlib import Path
from omegaconf import OmegaConf
from .cfg_templates import Arch, Data, Experiment, SbatchConfig, ExperimentsConfig


TRAIN_CMD_TEMPLATE = """srun --cpu_bind=none,v --accel-bind=gn python -u src/training/main.py --save-frequency=1 --dataset-type=webdataset --train-data={TRAIN_DATA} --train-num-samples={TRAIN_NUM_SAMPLES} --logs={LOGS} --warmup={WARMUP} --batch-size={BATCH_SIZE} --epochs={EPOCHS} --workers=4 --model {MODEL} --seed={SEED} --local-loss --gather-with-grad {DATASET_RESAMPLED} --log-every-n-steps=10 --coca-contrastive-loss-weight 1.0 --coca-caption-loss-weight 1.0 --report-to "wandb" --grad-checkpointing={GRAD_CHECKPOINTING} --lr={LR} --ddp-static-graph --precision={PRECISION} --name={RUN_NAME} --wandb-project-name={WANDB_PROJECT_NAME}  --val-frequency={VAL_FREQUENCY} --imagenet-val={IMAGENET_VAL_PATH}"""
VAL_CMD_TEMPLATE = """srun --cpu_bind=none,v --accel-bind=gn clip_benchmark eval --model={MODEL_NAME} --pretrained={PRETRAINED_PATH} --dataset={DATASET} --dataset_root={DATA_PATH} --output={OUTPUT_PATH} --batch_size={BATCH_SIZE}"""

train_cmd_template_kwargs = [
    "TRAIN_DATA", "TRAIN_NUM_SAMPLES", "LOGS", "WARMUP",
    "BATCH_SIZE", "EPOCHS", "MODEL", "SEED", "GRAD_CHECKPOINTING",
    "LR", "PRECISION", "WANDB_PROJECT_NAME", "VAL_FREQUENCY",
    "IMAGENET_VAL_PATH", "DATASET_RESAMPLED", "RUN_NAME",
]

val_cmd_template_kwargs = [
    "BATCH_SIZE", "DATA_PATH", "OUTPUT_PATH", "DATASET",
    "PRETRAINED_PATH", "MODEL_NAME"
]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --nodes={NODES}
#SBATCH --gpus-per-node={GPUS}
#SBATCH --account={ACCOUNT}
#SBATCH --time={TIME}
#SBATCH --ntasks-per-node={NTASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --partition={PARTITION}
#SBATCH --output={OUTPUT}
#SBATCH --job-name={JOB_NAME}
#SBATCH --array=1-{ARRAY}%{JOBS_AT_ONCE}
""" # %4 max number of jobs at the same time



sbatch_tempalte_kwargs = [
    "NODES", "GPUS", "ACCOUNT", "TIME",
    "NTASKS_PER_NODE", "CPUS_PER_TASK",
    "PARTITION", "OUTPUT", "JOB_NAME",
    "ARRAY"
]

def main(*, cfg, task="scripts", test=False, stage="train"):

    cfg = OmegaConf.load(cfg)
    # assert "models" in cfg
    # assert "datasets" in cfg
    # assert ""
    cfg = OmegaConf.structured(ExperimentsConfig(**cfg))

    experiments = []
    for model in cfg.models:
        for dataset in cfg.datasets:
            model_cfg = OmegaConf.structured(Arch(**model))
            dataset_cfg = OmegaConf.structured(Data(**dataset))

            experiments.append(
                OmegaConf.structured(
                    Experiment(**model_cfg, **dataset_cfg)
                )
            )

    sbatch_cfg = OmegaConf.structured(SbatchConfig(**cfg.sbatch_config))

    Path(sbatch_cfg.experiments_list_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sbatch_cfg.experiments_list_file_path, "w") as fd:
        for experiment in experiments:

            if experiment.stage == "train":
                cmd_template = TRAIN_CMD_TEMPLATE.format(
                    LR = experiment.lr,
                    TRAIN_DATA = experiment.data_path,
                    TRAIN_NUM_SAMPLES = experiment.size,
                    WARMUP = experiment.warmup,
                    BATCH_SIZE = experiment.batch_size,
                    EPOCHS = experiment.epochs,
                    MODEL = experiment.model_name,
                    DATASET_RESAMPLED = "--dataset-resampled" if experiment.resampled else "",
                    SEED = experiment.get("seed", None),
                    GRAD_CHECKPOINTING = "--grad-checkpointing" if experiment.checkpointing else "",
                    PRECISION = experiment.get("precision", None),
                    WANDB_PROJECT_NAME = experiment.get("wand_project_name", None),
                    RUN_NAME = "model_{}_data_{}_size_{}".format(experiment.model_name, experiment.data_name, experiment.size),
                    VAL_FREQUENCY = experiment.get("val_frequency", None),
                    IMAGENET_VAL_PATH = experiment.get("imagenet_val_path", None),
                    LOGS = experiment.get("logs", None),
                )

            elif experiment.stage == "val":
                cmd = VAL_CMD_TEMPLATE.format(
                    BATCH_SIZE = experiment.batch_size,
                    DATA_PATH = experiment.data_path,
                    OUTPUT_PATH = experiment.logs,
                    DATASET = experiment.data_name,
                    PRETRAINED_PATH = experiment.pretrained_path,
                    MODEL_NAME = experiment.model_name,
                )

            fd.write(cmd)
            fd.write("\n")


    tpl = SBATCH_TEMPLATE.format(
        NODES = sbatch_cfg.nodes,
        GPUS = sbatch_cfg.gpus,
        ACCOUNT = sbatch_cfg.account,
        TIME = sbatch_cfg.time,
        NTASKS_PER_NODE = sbatch_cfg.ntasks_per_node,
        CPUS_PER_TASK = sbatch_cfg.cpus_per_task,
        PARTITION = sbatch_cfg.partition,
        OUTPUT = sbatch_cfg.output,
        JOB_NAME = sbatch_cfg.job_name,
        TASKS_AT_ONCE= sbatch_cfg.jobs_at_once,
        ARRAY = len(experiments),
    )

    if test:
        tpl = SBATCH_TEMPLATE.format(
            NODES = 1,
            GPUS = 1,
            ACCOUNT = sbatch_cfg.account,
            TIME = "00:01:00",
            NTASKS_PER_NODE = 1,
            CPUS_PER_TASK = 1,
            PARTITION = "develbooster", # specific to juwels booster
            OUTPUT = sbatch_cfg.output,
            JOB_NAME = sbatch_cfg.job_name,
            ARRAY = 6
        )

    Path(sbatch_cfg.sbatch_script_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sbatch_cfg.sbatch_script_file_path, "w") as fd:
        fd.write(tpl)
        fd.write("\n\n\n")

        cmd = " \"$(cat {exp_list} | sed -n -e SLURM_ARRAY_TASK_IDp)\"".format(exp_list=sbatch_cfg.experiments_list_file_path)
        cmd = cmd.replace("SLURM_ARRAY_TASK_IDp", "\"${SLURM_ARRAY_TASK_ID}p\"")
        if not test:
            fd.write("srun --cpu_bind=none,v --accel-bind=gn python -u")
            fd.write(cmd)
        else:
            fd.write("mkdir -p test_logs\n")
            fd.write("echo")
            fd.write(cmd)
            fd.write(" >> test_logs/${SLURM_ARRAY_TASK_ID}_output.txt\n")
        fd.write("\necho MADEITTOTHEEND")


#                 elif task == "model_list":
#                     target_name = f"Model-{arch.name.replace('ViT-','')}_Data-{data.name}_Samples-{round(samples_seen/1e9)}B_lr-1e-3_bs-{int(global_bs//1e3)}k.pt"
#                     row = {}
#                     row['model_fullname'] = target_name
#                     row['arch'] = arch.name
#                     row['samples_per_epoch'], row['epochs'] = data_size, nb_epochs
#                     row['samples_seen'] = row['samples_per_epoch'] * row['epochs']
#                     #row['gmacs_per_sample'] = act[act.model==row['arch']].gmacs.values[0]
#                     #row['gmacs_total'] = row['samples_seen'] * row['gmacs_per_sample']
#                     row['upstream_dataset'] = f"LAION-{data.name}"
#                     rows.append(row)
#                     pd.DataFrame(rows).to_csv("openclip_meta_ext.csv", index=False)
#                     print(f"{arch.name},{target_name}")
#                 else:
#                     raise ValueError(task)
# main(cfg="experiments.yaml", test=True)