import os

import torch
from torch.utils.data import DataLoader
from src.encoder.VAE import BetaVAE
from VAE_train import MyCustomDataset


def test_reconstruction(model, dataloader, device, output_folder):
    model.eval()

    # folder = "data/std2.5_diff_vae_out_1"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # diff_out = "data/generation_ddm_64x0_unnorm_v2.pt"
    # diff_data = torch.load(diff_out)
    # for i in range(diff_data.shape[0]):
    #     with torch.no_grad():
    #         z = diff_data[i, :].to(device)
    #         z = torch.unsqueeze(z, dim=0)
    #         print(z.shape)
    #         new_reconstructed_data = model.decode(z)
    #     filename = f"{folder}/generate_{i + 81}.pt"
    #     torch.save(new_reconstructed_data, filename)

    # folder = "data/vae_sample"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # vae_sample = model.sample(5)
    # # print(torch.max(vae_sample))
    # # print(torch.min(vae_sample))
    # for i in range(vae_sample.shape[0]):
    #     vae_filename = f"{folder}/generate_{i + 11}.pt"
    #     tensor = vae_sample[i]
    #     # print(tensor.shape)
    #     tensor = tensor.unsqueeze(0)
    #     # print(tensor.shape)
    #     torch.save(tensor, vae_filename)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            print(data.shape)

            # Forward pass
            # out = model.forward(data)
            # reconstructed = out[0]
            # print(reconstructed.shape)

            # latent = model.get_latent(data)
            # filename = "data/std_2.5_latent_train_1.pt"
            # try:
            #     existing_data = torch.load(filename)
            #     existing_data.append(latent)
            #     torch.save(existing_data, filename)
            # except FileNotFoundError:
            #     torch.save([latent], filename)

            # Save original and reconstructed images
            # file_path_original = f"{output_folder}/original_{i+6}.pt"
            # file_path_reconstructed = f"{output_folder}/reconstructed_{i+6}.pt"
            # torch.save(data, file_path_original)
            # torch.save(reconstructed, file_path_reconstructed)

            if i >= 4:  # Save only 5 samples for testing
                break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = "data/fea_cat_small"
    output_folder = "data/std_2.5_weight_1"

    dataset = MyCustomDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    in_channels = 96  # Specify the number of input channels
    latent_dim = 1024  # Specify the size of the latent space
    kld_weight = 0.1  # Specify the weight for the KL divergence loss
    kl_std = 2.5
    beta_vae = BetaVAE(in_channels, latent_dim, kld_weight, kl_std=kl_std).to(device)
    checkpoint_dir = "vae_output/std_2.5_checkpoints_1"

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")  # Specify the path to the best model checkpoint
    beta_vae.load_state_dict(torch.load(best_model_path))
    # saved_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    # latest_checkpoint_path = os.path.join(checkpoint_dir, saved_checkpoints[-1])
    # checkpoint = torch.load(latest_checkpoint_path)
    # beta_vae.load_state_dict(checkpoint['model_state_dict'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    test_reconstruction(beta_vae, dataloader, device, output_folder)


if __name__ == "__main__":
    main()
