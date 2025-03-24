
import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import random
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import HFFST, CustomWriter
from datasets import STDataset
from utils import collate_fn, load_callbacks, load_config, load_loggers


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='her2st/TRIPLEX', help='logger path.')
    parser.add_argument('--gpu', type=int, default=[0], help='gpu id')
    parser.add_argument('--mode', type=str, default='train', help='train / test / external_test')
    parser.add_argument('--test_name', type=str, default='10x_breast_ff1', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3"}.')
    parser.add_argument('--exp_id', type=int, default=0, help='')
    parser.add_argument('--fold', type=int, default=0, help='')
    parser.add_argument('--model_path', type=str, default=r'', help='')

    args = parser.parse_args()
    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(cfg, fold=0):
    
    seed=cfg.GENERAL.seed
    name=cfg.MODEL.name
    data=cfg.DATASET.type
    batch_size=cfg.TRAINING.batch_size
    num_epochs=cfg.TRAINING.num_epochs
    mode = cfg.GENERAL.mode
    gpus = cfg.GENERAL.gpu
    exp_id = cfg.GENERAL.exp_id
    
    # Load dataset
    if mode == 'train':
        trainset = STDataset(mode='train', fold=fold, **cfg.DATASET)
        train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0, pin_memory=True, shuffle=True)
    
    if mode == 'external_test':
        testset = STDataset(mode=mode, fold=fold, test_data=cfg.GENERAL.test_name, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    
    else:
        testset = STDataset(mode='test', fold=fold, **cfg.DATASET)
        test_loader = DataLoader(testset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    
    log_name=f'{fold}-{name}-{data}-{seed}-{exp_id}'
    
    cfg.GENERAL.log_name = log_name
    
    # Load loggers and callbacks for Trainer
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)
    
    model_cfg = cfg.MODEL.copy()
    del model_cfg['name']
    
    # Load model
    print(model_cfg)
    model = HFFST(**model_cfg)
    
    # Train or test model
    if mode == 'train':
        # Instancialize Trainer 
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices = gpus,
            max_epochs = num_epochs,
            logger = loggers,
            check_val_every_n_epoch = 1,
            log_every_n_steps=10,
            callbacks = callbacks,
            amp_backend = 'native',
            precision = 16
        )
        
        trainer.fit(model, train_loader, test_loader)
        
    elif mode == 'external_test':
        trainer = pl.Trainer(accelerator="gpu", devices=gpus)
        
        checkpoint = cfg.GENERAL.model_path
        model = model.load_from_checkpoint(checkpoint, **model_cfg)
        
        trainer.test(model, test_loader)
        
    elif mode=='test':
        trainer = pl.Trainer(accelerator="gpu", devices=gpus)
        
        checkpoint = cfg.GENERAL.model_path
        model = model.load_from_checkpoint(checkpoint, **model_cfg)
        
        trainer.test(model, test_loader)
        
    else:
        raise Exception("Invalid mode")
    
    return model



if __name__ == '__main__':
    args = get_parse()
    cfg = load_config(args.config_name)

    seed = cfg.GENERAL.seed

    cfg.GENERAL.test_name = args.test_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    cfg.GENERAL.model_path = args.model_path
    cfg.GENERAL.mode = args.mode
    
    current_day = datetime.now().strftime('%Y-%m-%d')
    cfg.GENERAL.current_day = current_day
    
    if args.mode == 'train':
        num_k = cfg.TRAINING.num_k
        fix_seed(seed)
        for fold in range(6, num_k):
            main(cfg, fold=fold)
            
    else:
        main(cfg, args.fold)
