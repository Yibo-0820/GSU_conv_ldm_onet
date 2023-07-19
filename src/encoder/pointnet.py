import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from src.encoder.unet import UNet
from src.encoder.unet3d import UNet3D
from src.encoder.VAE import BetaVAE


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): whether to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, vae=False, vae_kwargs=None,
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5):
        super().__init__()
        self.vae_loss = None
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if vae:
            self.vae = BetaVAE(self.unet.conv_final.out_channels*3, **vae_kwargs)
        else:
            self.vae = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
            # print(fea_plane.shape)
            # filename = "data/fea_planes.pt"
            # with open(filename, "ab") as f:
            #     torch.save(fea_plane, f)
            # filename = "data/fea_planes_train_demo.pt"
            # try:
            #     existing_data = torch.load(filename)
            #     all_data = torch.cat((existing_data, fea_plane), dim=0)
            #     torch.save(all_data, filename)
            # except FileNotFoundError:
            #     torch.save(fea_plane, filename)
        #     try:
        #         existing_data = torch.load(filename)
        #         existing_data.append(fea_plane)
        #         torch.save(existing_data, filename)
        #     except FileNotFoundError:
        #         torch.save([fea_plane], filename)
        # if self.vae is not None:
        #     out = self.vae.forward(fea_plane)
        #     fea_plane = out[0]
        #     self.vae_loss = self.vae.loss_function(*out)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        original_fea = torch.cat((fea['xz'], fea['xy'], fea['yz']), dim=1)
        # mean = original_fea.mean()
        # std = original_fea.std()
        #
        # # Normalize the tensor
        # normalized_fea = (original_fea - mean) / std
        # print(normalized_fea)
        # print(original_fea.shape)

        # filename = "data/original_fea_train_6.pt"
        # try:
        #     existing_data = torch.load(filename)
        #     existing_data.append(original_fea)
        #     torch.save(existing_data, filename)
        # except FileNotFoundError:
        #     torch.save([original_fea], filename)
        if self.vae is not None:
            out = self.vae.forward(original_fea)
            fea_cat = out[0]
            # fea_cat = self.vae.sample(1)
            # latent = self.vae.get_latent(original_fea)
            # filename = "data/final_std_1_latent_train_1_new.pt"
            # try:
            #     existing_data = torch.load(filename)
            #     existing_data.append(latent)
            #     torch.save(existing_data, filename)
            # except FileNotFoundError:
            #     torch.save([latent], filename)
            self.vae_loss = self.vae.loss_function(*out)

            diff_out = "data/final/generation_1new.pt"
            diff_data = torch.load(diff_out)
            z = diff_data[11, :]
            z = torch.unsqueeze(z, dim=0)
            print(z.shape)
            new_reconstructed_data = self.vae.decode(z)
            fea_cat = new_reconstructed_data[0]
            fea_cat = torch.unsqueeze(fea_cat, dim=0)
            # print(fea_cat.shape)

        # fea_xz = fea[:, :32, :, :]
        #
        # # Retrieve fea['xy']
        # fea_xy = fea[:, 32:64, :, :]
        #
        # # Retrieve fea['yz']
        # fea_yz = fea[:, 64:, :, :]
        #
        # # Create the fea dictionary
        # fea = {'xz': fea_xz, 'xy': fea_xy, 'yz': fea_yz}

        '''generate'''
        # filename = "data/vae_sample/generate_11.pt"
        # filename = "data/std2.5_diff_vae_out_1/generate_82.pt"
        # filename = "data/std_2.5_weight_1/reconstructed_41.pt"
        # fea_all = torch.load(filename)
        # fea_cat = fea_all
        # fea_cat = fea_cat.squeeze(0)
        # print(fea_cat.shape)
        # print(fea_cat)
        fea_xz = fea_cat[:, :32, :, :]

        # Retrieve fea['xy']
        fea_xy = fea_cat[:, 32:64, :, :]

        # Retrieve fea['yz']
        fea_yz = fea_cat[:, 64:, :, :]

        # Create the fea dictionary
        fea_new = {'xz': fea_xz, 'xy': fea_xy, 'yz': fea_yz}

        return fea_new

class PatchLocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
        First transform input points to local system based on the given voxel size.
        Support non-fixed number of point cloud, but need to precompute the index
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        local_coord (bool): whether to use local coordinate
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        unit_size (float): defined voxel unit size for local system
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, 
                 local_coord=False, pos_encoding='linear', unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim

        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2*hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2*hidden_dim)

    def generate_plane_features(self, index, c):
        c = c.permute(0, 2, 1) 
        # scatter plane features from points
        if index.max() < self.reso_plane**2:
            fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane**2)
            fea_plane = scatter_mean(c, index, out=fea_plane) # B x c_dim x reso^2
        else:
            fea_plane = scatter_mean(c, index) # B x c_dim x reso^2
            if fea_plane.shape[-1] > self.reso_plane**2: # deal with outliers
                fea_plane = fea_plane[:, :, :-1]
        
        fea_plane = fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, index, c):
        # scatter grid features from points        
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid**3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
            fea_grid = scatter_mean(c, index, out=fea_grid) # B x c_dim x reso^3
        else:
            fea_grid = scatter_mean(c, index) # B x c_dim x reso^3
            if fea_grid.shape[-1] > self.reso_grid**3: # deal with outliers
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = index.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, inputs):
        p = inputs['points']
        index = inputs['index']
    
        batch_size, T, D = p.size()

        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(index['xz'], c)
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(index['xy'], c)
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(index['yz'], c)

        return fea
