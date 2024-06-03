# IRIS_MetaNet

## Run
#### Environment
* Python 3.7
* PyTorch 1.1.0
* torchvision 0.2.1
* [gpustat](https://github.com/wookayin/gpustat)

#### Train
##### Resnet-20 Models on CIFAR10
Run the script below and dataset will download automatically.
 ```
python train.py --cfg rn20_train_cfg
 ```

#### Test
##### Resnet-20 Models on CIFAR10
Run the script below and dataset will download automatically.
 ```
python train.py --cfg rn20_test_cfg
 ```

#### Config
##### Custom quanatization method
To add custom quantization method, define custom class in `models/quatizer.py` and use in config file.

``` python
# models/quantizer.py
class custom_quantization(torch.autograd.Function):
   # quantization logic
```
``` python
# config file
NN_cfg_dict = {
    "quant" : "custom_quantization"
}
```

#### Wandb
``` python
# config file
wandb_cfg_dict = {
    "wandb_enabled": True,
    "key": "YOUR_KEY",
    "entity": "YOUR_ENTITY",
    "project": "YOUR_PROJECT",
    "name": "WANDB_RUN_NAME",
}
```
If you want to name the wandb run, use `name` key. If this value is `None`, then run name will be wandb default name.

##### Use pretrained model from wandb
``` python
# config file
wandb_cfg_dict = {
    "pretrain": "wandb_run_path_to_download_pretrained_model",
}
```
If this value is `None`, then pretrained model will be loaded from `NN_cfg_dict["pretrain"]`.

##### Use sweep
``` python
# config file
wandb_cfg_dict = {
    "sweep_enabled": True,
    "sweep_config": {
        "name": NN_cfg_dict["log_id"],
        "method": "grid",
        "metric": {"goal": "maximize", "name": "Best_score"}
    },
    "sweep_count": 10000,
    "sweep_id": None
}
```
``` python
NN_cfg_dict = {
    "weight_bit_width": ["1,2,3,4,5,6,7,8,32", "1,2,4,8"],
}
```
Create a list with the parameters you want to experiment with.  
If you want to name the sweep, use `name` key in `sweep_config`. If this value is `None`, then name will be wandb default name.

## Reference
- [Any-Precision Deep Neural Network](https://github.com/SHI-Labs/Any-Precision-DNNs)