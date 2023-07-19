import torch
import argparse
from torch.utils.data import dataloader
from torch.utils.data import Dataset
from src.encoder.diffusion import DiffusionModel
from src.encoder.diffusion_arch import DiffusionNet
from src import config, data
from src.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']

model = DiffusionModel(model=DiffusionNet(**cfg["diffusion_model_specs"]), **cfg["diffusion_specs"]) 
# model.load_state_dict(torch.load('/usr/prakt/s0136/con_test/out/pointcloud/diff/model_10000.pt'))

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

model.eval()
model.to(device)

class MyCustomDataset(Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tensor = self.data[index]
        tensor = tensor.squeeze(0)
        return tensor


inputs = MyCustomDataset(filename='/usr/prakt/s0136/con_test/dataset/final_std_1_latent_train_50.pt')

val_loader = torch.utils.data.DataLoader(
        inputs, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


outputs = []
inp = []

with torch.no_grad():
    count = 0
    for batch in val_loader:
        input = batch
        input = input.to(device)
        inp.append(input)
        # t = torch.randint(0, model.num_timesteps, (1,), device=device).long()
        loss, diff_100_loss, diff_1000_loss, output, perturbed_pc = model.diffusion_model_from_latent(input, cond=None)
        # print(output[0])
        outputs.append(output)

        count += 1
    
        if count > 19:
            break

outputs = torch.cat(outputs, dim=0)
inp = torch.cat(inp, dim=0)
print(outputs.shape)
print(inp.shape)
print(inp)
print(outputs)

#generation
generation = model.generate_unconditional(num_samples=20)
print(generation)
print(generation.shape)

# print(outputs)
#torch.save(outputs, '/usr/prakt/s0136/con_test/out/pointcloud/diff_v4/output.pt')
torch.save(generation, '/usr/prakt/s0136/con_test/out/pointcloud/diff_final8_50/generation_50.pt')
#torch.save(inp, '/usr/prakt/s0136/con_test/out/pointcloud/diff_v4/inp.pt')
