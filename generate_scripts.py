from clize import run
import pandas as pd
import math
from pathlib import Path
import omegaconf
from cfg_templates import Arch, Data, Experiment


TEMPLATE = """srun --cpu_bind=none,v --accel-bind=gn python -u src/training/main.py --save-frequency=1 --dataset-type=webdataset --train-data={TRAIN_DATA} --train-num-samples={TRAIN_NUM_SAMPLES} --logs={LOGS} --warmup={WARMUP} --batch-size={BATCH_SIZE} --epochs={EPOCHS} --workers=4 --model {MODEL} --seed={SEED} --local-loss --gather-with-grad {DATASET_RESAMPLED} --log-every-n-steps=10 --coca-contrastive-loss-weight 1.0 --coca-caption-loss-weight 1.0 --report-to "wandb" --grad-checkpointing={GRAD_CHECKPOINTING} --lr={LR} --ddp-static-graph --precision={PRECISION} --wandb-project-name={WANDB_PROJECT_NAME}  --val-frequency={VAL_FREQUENCY} --imagenet-val={IMAGENET_VAL_PATH}"""

template_kwargs = [
    "TRAIN_DATA", "TRAIN_NUM_SAMPLES", "LOGS", "WARMUP",
    "BATCH_SIZE", "EPOCHS", "MODEL", "SEED", "GRAD_CHECKPOINTING",
    "LR", "PRECISION", "WANDB_PROJECT_NAME", "VAL_FREQUENCY",
    "IMAGENET_VAL_PATH", "DATASET_RESAMPLED"
]

def main(*, cfg, task="scripts"):

    cfg = omegaconf.OmegaConf.load(cfg)
    assert "models" in cfg
    assert "datasets" in cfg

    experiments = []
    for model in cfg.models:
        for dataset in cfg.datasets:
            model_cfg = omegaconf.OmegaConf.structured(Arch(**model))
            dataset_cfg = omegaconf.OmegaConf.structured(Data(**dataset))

            experiments.append(
                omegaconf.OmegaConf.structured(
                    Experiment(**model_cfg, **dataset_cfg)
                )
            )

    Path(cfg.experiments_list_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.experiments_list_file, "w") as fd:
        for experiment in experiments:

            if experiment.data_done and experiment.model_done:
                continue

            cmd = TEMPLATE.format(
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
                VAL_FREQUENCY = experiment.get("val_frequency", None),
                IMAGENET_VAL_PATH = experiment.get("imagenet_val_path", None),
                LOGS = experiment.get("logs", None),
            )

            fd.write(cmd)

    with open(cfg.sbatch_script_template_path) as fd:
        tpl = fd.read()

    Path(cfg.sbatch_script_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.sbatch_script_path, "w") as fd:
        fd.write(tpl)
        fd.write("\n")
        fd.write("srun --cpu_bind=none,v --accel-bind=gn python -u")
        fd.write(" \"$(cat {exp_list} | sed -n -p SLURM_ARRAY_TASK_IDp)\"".format(exp_list=cfg.experiments_list_file))

#     delta = 0
#     act = pd.read_csv('clip_table_2.csv')
#     act['vit'] = act.model.str.contains('ViT')
#     act = act[act.vit]
#     rows = []
#     for arch_type in archs.keys():
#         for arch_name, arch in arch_type.items():
#             for dataset in datasets:
#                 for data_name, data in dataset.items():

#                     if data.resampled:
#                         nb_epochs = (samples_seen) // (200e6)
#                         data_size = int(samples_seen // nb_epochs)
#                         nb_epochs = (samples_seen // data_size)
#                     else:
#                         data_size = data.size
#                         nb_epochs1 = math.ceil(samples_seen / data.size)
#                         nb_epochs2 = (samples_seen // data.size)
#                         if abs(nb_epochs1 * data_size - samples_seen) < abs(nb_epochs2 * data_size - samples_seen):
#                             nb_epochs = nb_epochs1
#                         else:
#                             nb_epochs = nb_epochs2

#                 nb_nodes = arch.nodes
#                 if target in ("boosterv2", "stability2v3"):
#                     global_bs = 32768
#                     arch.lr = 5e-4
#                 else:
#                     global_bs = 88_000
#                 nb_nodes = math.ceil(global_bs / (arch.bs * gpus_per_node))
#                 global_bs = arch.bs * gpus_per_node * nb_nodes
#                 name = f"{arch.name}_{data.name}_{int(samples_seen/1e9)}b"
#                 # print(name, global_bs, nb_nodes, samples_seen/1e9, data.name, arch.name, arch.bs, nb_epochs)
#                 delta += abs(nb_epochs*data_size-samples_seen)/1e6
#                 script = tpl.format(
#                     ACCOUNT=account,
#                     NODES=nb_nodes,
#                     NAME=name,
#                     DATA=data.path,
#                     RESAMPLED=data.resampled_str,
#                     WARMUP=arch.warmup,
#                     MODEL=str(arch),
#                     BATCH_SIZE=arch.bs,
#                     EPOCHS=nb_epochs,
#                     NUM_SAMPLES=data_size,
#                     LR=arch.lr,
#                 )
#                 if task == "scripts":
#                     print(name, nb_epochs * data_size, samples_seen,  abs(nb_epochs*data_size-samples_seen)/1e6  )
#                     with open(target + "/" + name+".sbatch", "w") as fd:
#                         fd.write(script)
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
# run(main)
main(cfg="experiments.yaml")