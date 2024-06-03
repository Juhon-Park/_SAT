import os
import torch
import logging
import shutil
import gpustat
import random
from datetime import datetime
from misc.optimizer import get_optimizer_config, get_lr_scheduler
from configs.config import *
import wandb

args = all_cfg


class AverageMeter:
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.float().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def setup_gpus():
    """Adapted from https://github.com/bamos/setGPU/blob/master/setGPU.py
    """
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    best_gpu = min(pairs, key=lambda x: x[1])[0]
    return best_gpu


def save_checkpoint(state, is_best, path, name='model_latest.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = path + '/' + name
    torch.save(state, save_path)
    logging.info('checkpoint saved to {}'.format(save_path))
    if is_best:
        shutil.copyfile(save_path, path + '/model_best.pth.tar')

def results_dir_config(train_id, model_name):

    train_id = train_id
    train_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_id = f"{train_id}_{train_stamp}"
    curr_dir = os.getcwd()

    if "resnet20" in model_name: 
        model_dir = "RN20"
    elif "resnet50" in model_name:
        model_dir = "RN50"
    elif "resnet18" in model_name:
        model_dir = "RN18"
    else: 
        model_dir = ""
    
    if ("debug" in train_id) or ("test" in train_id):
        model_dir = "debug"

    result_dir = os.path.join("/workspace/IRIS_MetaNet_ws", "results", model_dir, train_id)
    os.makedirs(result_dir, exist_ok=True)

    return result_dir
    

def check_resume_pretrain(model, optimizer, best_gpu):
    lr_scheduler = None
    best_prec1 = None
    lr_decay = list(map(int, args.lr_decay.split(',')))
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(best_gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.scheduler, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    elif wandb_cfg.pretrain:
        api = wandb.Api(overrides={"entity": wandb_cfg.entity})
        runs = api.run(wandb_cfg.pretrain)
        for artifact in runs.logged_artifacts():
            if "best" in artifact.name:
                arti = runs.use_artifact(artifact)
                arti.download(root="./pretrained", path_prefix="model_best.pth.tar")
                arti.wait()
                break
        checkpoint = torch.load("./pretrained/model_best.pth.tar", map_location='cuda:{}'.format(best_gpu))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded pretrain checkpoint '%s' (epoch %s)", wandb_cfg.pretrain, checkpoint['epoch'])
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')

    return best_prec1, lr_scheduler

def freeze_param(model, cal_bw):
    if (args.is_training == 'T' and args.is_calibrate == 'T'):
        calibrate = ["bn_dict." + str(i) + "." for i in cal_bw]
        for (name, param) in model.named_parameters():
            if any(x in name for x in calibrate):
                param.requires_grad = True
            else:
                param.requires_grad = False

def save_ckpt(epoch, model, best_prec1, optimizer, is_best, path, name='model_latest.pth.tar'):
    ckpt_dict = {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                }
    
    save_checkpoint(ckpt_dict, is_best, path=path)