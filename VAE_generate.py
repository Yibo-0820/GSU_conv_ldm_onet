import torch
from src.encoder.VAE import BetaVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model state
model_path = "beta_vae_2.pt"
in_channels = 96  # Specify the number of input channels
latent_dim = 1024  # Specify the size of the latent space
kld_weight = 0.1  # Specify the weight for the KL divergence loss
kl_std = 0.25
beta_vae = BetaVAE(in_channels, latent_dim, kld_weight, kl_std=kl_std).to(device)
beta_vae.load_state_dict(torch.load(model_path))
beta_vae.eval()

filename = "data/original_fea_train_demo.pt"
diff_out = "data/output.pt"

# Prepare your test data
test_data = torch.load(filename)
diff_data = torch.load(diff_out)

# Pass the test data through the model
# with torch.no_grad():
#     for data in test_data:
#     # print(len(test_data))
#         data = data.to(device)
#         # reconstructed_data = beta_vae(data)
#         latent = beta_vae.get_latent(data)
#
#         filename = "data/latent_demo.pt"
#         try:
#             existing_data = torch.load(filename)
#             existing_data.append(latent)
#             torch.save(existing_data, filename)
#         except FileNotFoundError:
#             torch.save([latent], filename)

# filename_2 = "data/original_fea_test_result_2.pt"
# torch.save([reconstructed_data], filename_2)

# filename_3 = "data/latent_demo.pt"
# torch.save(latent, filename_3)

with torch.no_grad():
    z = diff_data[0, :].to(device)
    new_reconstructed_data = beta_vae.decode(z)
filename_4 = "data/diff_vae_out_demo.pt"
torch.save(new_reconstructed_data, filename_4)
