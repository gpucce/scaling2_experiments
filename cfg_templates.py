from dataclasses import dataclass

@dataclass
class Data:
    data_name: str
    data_path: str
    size: int
    resampled: bool
    epochs: int
    data_done: bool = False



@dataclass
class Arch:
    model_name: str
    batch_size: int
    warmup: int
    lr: float
    checkpointing: bool
    nodes: int
    precision: str = "amp_bfloat16"

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
    data_done: bool = False

