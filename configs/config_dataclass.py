import os
from dataclasses import dataclass, fields, make_dataclass, asdict
from typing import get_type_hints, Any, Dict, List, Tuple
from icecream import ic

def validate_types(my_object):
    type_hints = get_type_hints(my_object.__class__)
    for field_name, field_type in type_hints.items():
        value = getattr(my_object, field_name)
        if isinstance(value, list):
            if not all(isinstance(item, field_type) for item in value):
                raise ValueError(f"All elements of '{field_name}' must be of type {field_type}")
        else:
            if not isinstance(value, field_type) and value is not None:
                raise TypeError(f"Field '{field_name}' expected type {field_type}, got value {value} of type {type(value)}")


def set_all_fields_to_none(my_object):
    for field in fields(my_object)[1:]:
        setattr(my_object, field.name, None)


def combine_dc_to_dict(use_sweep: bool, dataclass_list: list) -> Dict[str, Any]:
    combined_dict = {}
    for dc in dataclass_list:
        curr_dict = asdict(dc)
        if not use_sweep:
            for key, value in curr_dict.items():
                if isinstance(value, list):
                    raise ValueError(f"Field '{key}' expected type {type(value[0])}, got type {type(value)}")
        combined_dict = {**combined_dict, **curr_dict}
    return combined_dict


def create_dc_from_dict(name, data_dict):
    return make_dataclass(name, data_dict.items())


@dataclass
class NN_config:
    log_id: str = 'RN20_test'
    dataset: str = 'cifar10'
    train_split: str = 'train'
    model: str = 'resnet20q'
    workers: int = 2
    epochs: int = 200
    start_epoch: int = 0
    batch_size: int = 512
    optimizer: str = 'sgd'
    scheduler: str = 'multi'
    lr: float = 0.1
    lr_decay: str = '100,150,180'
    weight_decay: float = 5e-4
    print_freq: int = 20
    pretrain: str = None
    resume: str = None
    weight_bit_width: str = 4
    act_bit_width: str = 4
    is_training: str = 'F'
    is_calibrate: str = 'F'
    cal_bit_width: str = 4
    quant: str = 'quant_dorefa'
    CONV: str = 'conv2dQ'

    def __post_init__(self):
        validate_types(self)
        ic(asdict(self))
    

@dataclass
class LR_config:
    LR_enabled: bool = True
    rank: int = 1
    groups: int = 1

    def __post_init__(self):
        validate_types(self)
        if not self.LR_enabled:
            # set all fields to None
            set_all_fields_to_none(self)
        ic(asdict(self))


@dataclass
class wandb_config:
    wandb_enabled: bool
    key: str
    entity: str
    project: str
    sweep_enabled: bool
    sweep_config: dict
    sweep_count: int
    name: str = None
    pretrain: str = None
    sweep_id: str = None

    def __post_init__(self):
        validate_types(self)
        if not self.sweep_enabled:
            self.sweep_config = None
            self.sweep_count = None
            self.sweep_id = None
        if self.sweep_id is not None:
            self.sweep_config = None
            self.sweep_count = None
        ic(asdict(self))
