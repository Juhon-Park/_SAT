import argparse
import importlib

__all__ = ["wandb_cfg", "LR_cfg", "NN_cfg", "all_cfg", "cfg_module", "layer_cfg"]

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--cfg', default='rn20_test_cfg', help='choose config file')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
cfg_args = parser.parse_args()
cfg_file = f"configs.{cfg_args.cfg}"
cfg_module = importlib.import_module(cfg_file)
module_attributes = vars(cfg_module)
globals().update({name: value for name, value in module_attributes.items() if not name.startswith('_')})

NN_cfg = NN_config(**NN_cfg_dict)
LR_cfg = LR_config(**LR_cfg_dict)
wandb_cfg = wandb_config(**wandb_cfg_dict)

comb_dict = combine_dc_to_dict(wandb_cfg.sweep_enabled, [NN_cfg, LR_cfg])
all_cfg_dc = create_dc_from_dict("all_cfg_dc", comb_dict)

all_cfg = all_cfg_dc(**comb_dict)

layer_cfg = layer_cfg_dict