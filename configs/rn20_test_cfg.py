from configs.config_dataclass import *

__all__ = ["wandb_cfg_dict", "LR_cfg_dict", "NN_cfg_dict", "layer_cfg_dict"]

NN_cfg_dict = {
    "log_id": "hello",
    "dataset": "cifar10",
    "train_split": "train",
    "model": "preresnet20q",
    "workers": 2,
    "epochs": 1,
    "start_epoch": 0,
    "batch_size": 128,
    "optimizer": "adam",
    "scheduler": "multi",
    "lr": 0.001,
    "lr_decay": "100,140,180,220",
    "weight_decay": 0.0,
    "print_freq": 20,
    "pretrain": "/workspace/MetaNet_ws/pretrain/RN20T_124832_3567_exp1.pth.tar",
    "resume": None,
    "weight_bit_width": "1,2,3,4,5,6,7,8,32",
    "act_bit_width": "4",
    "is_training": 'F',
    "is_calibrate": 'F',
    "cal_bit_width": "1,2,3,4,5,6,7",
    "quant": "truncquant_inference",
    "CONV": "conv2dQ"
}

LR_cfg_dict = {
    "LR_enabled": False,
    "rank": 4,
    "groups": 8
}

wandb_cfg_dict = {
    "wandb_enabled": True,
    "key": "028a6c9f793dd46f8ead875b50784dde31c413be",
    "entity": "iris_metanet",
    "project": "hello",
    "name": "hello",
    "pretrain": None,
    "sweep_enabled": False,
    "sweep_config": {
        "name": NN_cfg_dict["log_id"],
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}

layer_cfg_dict = {
    "layer1.0.conv0": "conv2dQ_lr",
    "layer1.1.conv1": "conv2dQ_lr"
}
