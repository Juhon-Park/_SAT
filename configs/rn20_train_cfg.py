from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "train",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "resnet20q",
    "workers":2,
    "epochs": 130,
    "start_epoch": 0,
    "batch_size": 8,
    "optimizer": "adam",
    "scheduler": "multi",
    "lr": 0.001,
    "lr_decay": "100,140,180,220",
    "weight_decay": 0.0,
    "print_freq": 20,
    "pretrain": None,
    "resume": None,
    "weight_bit_width": "2,4,8,32",
    "act_bit_width": "4,4,4,32",
    "is_training": 'T',
    "is_calibrate": 'F',
    "cal_bit_width": "1,2,3,4,5,6,7",
    "quant" : "truncquant",
    "CONV": "conv2d_sparse"
}

LR_cfg_dict = {
    "LR_enabled": False,
    "rank": 4,
    "groups": 4
}

wandb_cfg_dict = {
    "wandb_enabled": False,
    "key": "463b69d4a0864d56d8a99e38d67a896af404b7cc",
    "entity": "heesungjeong",
    "project": "resnet8q",
    "sweep_enabled": False,
    "sweep_config": {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}

layer_cfg_dict = {
    "layer1.0.conv0": "conv2d_noQ",
    "layer1.0.conv1": "conv2dQ_lr"
    
}