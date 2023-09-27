from dataclasses import dataclass

@dataclass
class Data:
    data_name: str
    data_path: str
    size: int
    resampled: bool
    data_done: bool = False



@dataclass
class Arch:
    model_name: str
    batch_size: int
    warmup: int
    lr: float
    checkpointing: bool
    nodes: int

@dataclass
class Experiment:
    model_name: str
    batch_size:int
    warmup: int
    lr: float
    checkpointing: bool
    nodes: int
    data_name: str
    data_path: str
    size: int
    resampled: bool

