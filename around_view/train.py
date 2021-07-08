""" Training routine for AroundView ViewSelector model. 
    modified from graspnet baseline train.py
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from around_view.utils.dataset import AroundViewFeatureDataset as AroundViewDataset
from around_view.utils.dataset import collate_fn_avf as collate_fn
from around_view.models.rnn import RNNController
from around_view.models.loss import LossComputer
from around_view.models.pytorch_utils import device

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--method', required=True, help='the method of selecting views')
parser.add_argument('--num_workers', default=4, help='num_workers of dataloader')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_view', type=int, default=5, help='view index: [0, 256)')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)
log_dir = os.path.join(cfgs.log_dir, str(datetime.now())[:-7].replace(':', '_').replace(' ', '-'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
LOG_FOUT = open(os.path.join(log_dir, f'log.txt'), 'a')
LOG_FOUT.write('Configs:\n\t' + str(cfgs).replace(' ', '\n\t')[10:-1]+'\n')
def log_string(out_str):
    str_time = str(datetime.now()).split('.')[0]
    out_str = f'[{str_time}] {out_str}'
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

DEFAULT_CHECKPOINT_PATH = os.path.join(log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
TRAIN_DATASET = AroundViewDataset(cfgs.dataset_root, camera=cfgs.camera, split='train', num_points=cfgs.num_point, augment=True)
TEST_DATASET = AroundViewDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, augment=False)
print(f'len(train data) = {len(TRAIN_DATASET)};  len(test data): {len(TEST_DATASET)}')
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=int(cfgs.num_workers), worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=int(cfgs.num_workers), worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(f'len(train loader) = {len(TRAIN_DATALOADER)};  len(test loader): {len(TEST_DATALOADER)}')

# Init the model and optimzier
net = RNNController(cfgs=cfgs, device=device)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
losser = LossComputer(cfgs)

# Load checkpoint if there is any
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch, stat_dict):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    stat_dict[f'{losser.author_tag}/lr'] = lr
    return stat_dict

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(log_dir, 'test'))
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def accumulate_statistics(infos, stat_dict):
    # Accumulate statistics and print out
    for key in infos:
        if 'loss' in key or 'mAP' in key or 'reward' in key:
            if key not in stat_dict: stat_dict[key] = 0
            stat_dict[key] += infos[key]
    return stat_dict

def log_statistics(tag_str, interval, stat_dict, writer, step, log_str=''):
    log_str += f'({tag_str}) '
    for key in sorted(stat_dict.keys()):
        value = round(1.0 * stat_dict[key] / interval, 8)
        writer.add_scalar(key, value, step)
        log_str += f'{key}: {value}, '
        stat_dict[key] = 0
    log_string(log_str)

def train_one_epoch():
    stat_dict = {} # collect statistics
    stat_dict = adjust_learning_rate(optimizer, EPOCH_CNT, stat_dict)
    # set model to training mode
    net.train()
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        batch_data['cloud_feats'] = batch_data['cloud_feats'].to(device)
        end_views = net(batch_data)
        loss, infos = losser.get_loss(end_views)
        loss.backward()
        if (batch_idx+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
        stat_dict = accumulate_statistics(infos, stat_dict)

        interval = 10
        if (batch_idx+1) % interval == 0:
            step = (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size
            log_str = f'({batch_idx+1}/{len(TRAIN_DATALOADER)})'
            log_statistics('train', 10, stat_dict, TRAIN_WRITER, step, log_str)

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        batch_data['cloud_feats'] = batch_data['cloud_feats'].to(device)
        with torch.no_grad():
            end_views = net(batch_data)
        loss, infos = losser.get_loss(end_views)
        stat_dict = accumulate_statistics(infos, stat_dict)

    interval = len(TEST_DATALOADER)
    step = (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size
    log_statistics('val', interval, stat_dict, TEST_WRITER, step)
    mean_loss = 1.0 * stat_dict[f'{losser.author_tag}/loss'] / interval
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('******** EPOCH %03d ********' % (epoch+1))
        # log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(log_dir, f'epoch_{EPOCH_CNT}.tar'))

if __name__=='__main__':
    train(start_epoch)
