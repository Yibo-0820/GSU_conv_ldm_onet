import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.encoder.VAE import BetaVAE


class MyCustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = sorted(os.listdir(data_folder))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_folder, self.file_list[index])
        tensor = torch.load(file_path)
        tensor = tensor.squeeze(0)
        return tensor


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save loss to a file
    loss_file_path = os.path.join(checkpoint_dir, "loss.txt")
    with open(loss_file_path, 'a') as f:
        f.write(f"Epoch {epoch}: Loss={loss}\n")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def save_best_model(model, best_loss, checkpoint_dir):
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(model.state_dict(), best_model_path)

    # Save best loss to a file
    loss_file_path = os.path.join(checkpoint_dir, "loss.txt")
    with open(loss_file_path, 'a') as f:
        f.write(f"Best Model: Loss={best_loss}\n")


def train(model, dataloader, optimizer, device, checkpoint_dir):
    total_loss = 0
    best_loss = float('inf')
    epoch = 0

    # Check if there is a best model checkpoint, get the best loss

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.train()
        with open(os.path.join(checkpoint_dir, "loss.txt"), 'r') as f:
            for line in f:
                if line.startswith('Best Model:'):
                    best_loss = line.split('Loss=')[1].strip()
                    best_loss = float(best_loss)

        # Find the last saved checkpoint, get the epoch
        saved_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        if saved_checkpoints:
            last_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in saved_checkpoints])
            epoch = last_epoch

    while True:
        epoch += 1
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            model.train()

            # Forward pass
            out = model.forward(data)

            # Compute the reconstruction loss
            recon_loss = F.mse_loss(out[0], data)

            # Compute the KL divergence loss
            kl_loss = model.loss_function(*out)

            # Total loss
            loss = recon_loss + kl_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}")
        print(f"Loss: {average_loss:.4f}")
        total_loss = 0

        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, epoch, average_loss, checkpoint_dir)

        # Update best model
        if average_loss < best_loss:
            best_loss = average_loss
            save_best_model(model, best_loss, checkpoint_dir)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "data/fea_cat_small"

    dataset = MyCustomDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    in_channels = 96  # Specify the number of input channels
    latent_dim = 1024  # Specify the size of the latent space
    kld_weight = 0.5  # Specify the weight for the KL divergence loss
    kl_std = 1
    beta_vae = BetaVAE(in_channels, latent_dim, kld_weight, kl_std=kl_std).to(device)

    # Set up your optimizer
    learning_rate = 0.00005  # Specify the learning rate
    optimizer = optim.Adam(beta_vae.parameters(), lr=learning_rate)

    checkpoint_dir = "vae_output/std_1_checkpoints_0.5"  # Specify the directory to store checkpoints

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    saved_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if len(saved_checkpoints) != 0:
        latest_checkpoint_path = os.path.join(checkpoint_dir, saved_checkpoints[-1])
        if os.path.exists(latest_checkpoint_path):
            print("Loading latest model checkpoint...")
            beta_vae, optimizer, _, _ = load_checkpoint(beta_vae, optimizer, latest_checkpoint_path)
    else:
        if os.path.exists(best_model_path):
            print("Loading best model checkpoint...")
            beta_vae.load_state_dict(torch.load(best_model_path))

    try:
        while True:
            train(beta_vae, dataloader, optimizer, device, checkpoint_dir)

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        # Save the current model
        interrupt_point = os.path.join(checkpoint_dir, "beta_vae_interrupted.pt")
        torch.save(beta_vae.state_dict(), interrupt_point)
        sys.exit(0)


if __name__ == "__main__":
    main()
