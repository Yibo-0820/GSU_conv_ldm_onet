from statistics import mean
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib
from diff_utils.helpers import load_model, perturb_point_cloud, save_model; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import shutil
from tqdm import tqdm

from src.encoder.diffusion import DiffusionModel
from src.encoder.diffusion_arch import DiffusionNet

parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")    
    
class Trainer:
    def __init__(self, model, optimizer,batch_size):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train_step(self, batch):
        inputs = batch
        targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        self.optimizer.zero_grad()
        t = torch.randint(0, self.model.num_timesteps, (self.batch_size,), device=device).long()
        #outputs = self.model(inputs,t)
        # loss, xt, target, pred, unreduced_loss = self.model(inputs, t, ret_pred_x=True)

        loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.model.diffusion_model_from_latent(inputs, cond=None)


        loss.backward()
        self.optimizer.step()

        # return loss.item()
        return loss

    def evaluate(self, data_loader):
        eval_dict = defaultdict(float)
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch
                targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                t = torch.randint(0, self.model.num_timesteps, (self.batch_size,), device=device).long()

                loss, diff_100_loss, diff_1000_loss, outputs, perturbed_pc = self.model.diffusion_model_from_latent(inputs, cond=None)
                batch_size = outputs.size(0)

                # Accumulate evaluation metrics
                eval_dict['mse'] += loss.item() * batch_size
                total_samples += batch_size

        # Average evaluation metrics over all samples
        for k in eval_dict:
            eval_dict[k] /= total_samples

        self.model.train()

        return eval_dict