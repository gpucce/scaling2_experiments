from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Data:
    data_name: str
    data_path: str
    size: int
    resampled: bool
    epochs: int
    data_done: Optional[bool] = False
    imagenet_val_path: Optional[str] = None
    val_frequency: Optional[int] = None

@dataclass
class Arch:
    model_name: str
    batch_size: int
    warmup: int
    lr: float
    checkpointing: bool
    nodes: int
    precision: str = "amp_bfloat16"
    seed: Optional[int] = None
    wandb_project_name: Optional[str] = None
    logs: Optional[str] = None


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

@dataclass
class SbatchConfig:
    nodes: int
    gpus: int
    account: str
    time: str
    ntasks_per_node: int
    cpus_per_task: int
    partition: str
    output: str
    job_name: str
    experiments_list_file_path: str
    sbatch_script_file_path: str


@dataclass
class ExperimentsConfig:
    models: List[Arch]
    datasets: List[Data]
    sbatch_config: SbatchConfig