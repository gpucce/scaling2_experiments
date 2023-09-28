from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Data:
    data_name: str
    data_path: str
    size: int = 1
    epochs: int = 1
    resampled: bool = True
    data_done: Optional[bool] = False
    imagenet_val_path: Optional[str] = None
    val_frequency: Optional[int] = None
    task: Optional[str] = None
    distributed: bool = True

@dataclass
class Arch:
    model_name: str
    pretrained_path: Optional[str] = None
    batch_size: int = 4
    lr: float = 0.001
    warmup: int = 0
    checkpointing: bool = True
    nodes: int = 1
    precision: str = "amp_bfloat16"
    seed: Optional[int] = None
    wandb_project_name: Optional[str] = None
    logs: Optional[str] = None
    stage: Optional[str] = "train"

@dataclass
class Experiment:
    model_name: str
    batch_size:int
    warmup: int
    lr: float
    checkpointing: bool
    nodes: int
    precision: str
    data_name: str
    data_path: str
    size: int
    epochs: int
    resampled: bool
    data_done: bool
    seed: Optional[int]
    wandb_project_name: Optional[str]
    imagenet_val_path: Optional[str]
    logs: Optional[str]
    val_frequency: Optional[int]
    pretrained_path: Optional[str]
    task: Optional[str]
    stage: Optional[str]
    distributed: bool

@dataclass
class SbatchConfig:
    account: str
    partition: str
    nodes: int = 1
    gpus: int = 1
    time: str = "00:30:00"
    ntasks_per_node: int = 1
    cpus_per_task: int = 6
    output: str = "./output"
    job_name: str = "automated_experiments"
    experiments_list_file_path: str = "experiments_list.txt"
    sbatch_script_file_path: str = "./sbatch_script.sbatch"
    jobs_at_once: int = 4
    extra_preamble: List[str] = None


@dataclass
class ExperimentsConfig:
    models: List[Arch]
    datasets: List[Data]
    sbatch_config: SbatchConfig