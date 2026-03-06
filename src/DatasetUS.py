import torch
import torchvision
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import Transformer
from datetime import datetime
from torch.masked import masked_tensor

class UpscaleDatasetCERRA(torch.utils.data.Dataset):
    """
    Dataset class of images with a low resolution and a high resolution counterpart
    from CERRA and ERA5 datasets.
    """

    def __init__(self, data_dir,
                 year_start=1985, year_end=2008,
                 normalize_rawdata_mean=torch.Tensor([283.3926]),
                 normalize_rawdata_std=torch.Tensor([7.3598]),
                 normalize_residual_mean=torch.Tensor([-.1746]),
                 normalize_residual_std=torch.Tensor([1.2076]),
                 constant_variables=["lsm", "orog"],
                 constant_variables_path= os.path.join(os.path.dirname(__file__),
                                                       "../data/CERRA-ERA5/cerra_const_sfc_variables.nc4c"),
                 sea_masking=False
                 ):
        """
        :param data_dir: path to the dataset directory
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
        """

        print("Opening CERRA files")

        self.filenames_c = [f"cerra_samples_{year}.nc4c" for year in range(year_start, year_end+1)]

        # Open first file for saving projection info
        filename0 = self.filenames_c[0]
        path_to_file = data_dir + filename0
        ds_c = xr.open_dataset(path_to_file, engine="netcdf4")

        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 lon/lat
            "+proj=lcc +lat_0=50 +lon_0=8 +lat_1=50 +lat_2=50 +R=6371229 +units=m",
            always_xy=True
        )

        lon0 = ds_c['Lambert_Conformal'].attrs['longitudeOfFirstGridPointInDegrees']
        lat0 = ds_c['Lambert_Conformal'].attrs['latitudeOfFirstGridPointInDegrees']

        x0, y0 = transformer.transform(lon0, lat0)

        self.projection = ccrs.LambertConformal(
            central_longitude=8,
            central_latitude=50,
            standard_parallels=(50, 50),
            globe=ccrs.Globe(ellipse='sphere', semimajor_axis=6371229.0),
            false_easting=-x0,
            false_northing=-y0
        )

        self.x = ds_c.x  # len 256
        self.y = ds_c.y  # len 256
        self.nx = self.W = len(self.x)  # Width
        self.ny = self.H = len(self.y)  # Height

        self.varnames = ["temp"]
        self.n_var = len(self.varnames)

        # Concatenate other files
        for filename in self.filenames_c[1:]:
            path_to_file = data_dir + filename
            ds2 = xr.open_dataset(path_to_file, engine="netcdf4")
            ds_c = xr.concat((ds_c, ds2), dim="time")

        self.ntime = len(ds_c.time)

        print("All files accessed. Creating tensors")

        # Convert xarray dataarrays into torch Tensor (loads into memory)
        t = torch.from_numpy(ds_c['2t'].to_numpy()).float()

        # Stack into (ntime, 1, 256, 256), creating the fine resolution image.
        cerra = t #.unsqueeze(1)
        print('CERRA tensor:', cerra.shape)

        print("Opening ERA files")

        self.filenames_e = [f"era5_samples_{year}.nc4c" for year in range(year_start, year_end+1)]

        # Open first file for saving dimension info
        filename0 = self.filenames_e[0]
        path_to_file = data_dir + filename0
        ds_e = xr.open_dataset(path_to_file, engine="netcdf4")

        # self.coarse_x = ds_e.x  # len 256
        # self.coarse_y = ds_e.y  # len 256

        # Concatenate other files
        for filename in self.filenames_e[1:]:
            path_to_file = data_dir + filename
            ds2 = xr.open_dataset(path_to_file, engine="netcdf4")
            ds_e = xr.concat((ds_e, ds2), dim="time")

        # Convert xarray dataarrays into torch Tensor (loads into memory)
        t = torch.from_numpy(ds_e['2t'].to_numpy()).float()

        # Stack into (ntime, 1, 64, 64), creating the coarse resolution image.
        era5 = t.unsqueeze(1)
        print('ERA5 tensor:', era5.shape)

        # Calculate residual = fine - coarse. this will be our target
        residual = cerra - era5

        # Save unnormalized coarse and fine images for plotting
        self.coarse = era5
        self.fine = cerra

        # Normalize : use raw data means for coarse image
        normalize_rawdata_transform = torchvision.transforms.Normalize(normalize_rawdata_mean, normalize_rawdata_std)
        coarse_norm = normalize_rawdata_transform(era5)

        # use residual means for the difference between them
        normalize_residual_transform = torchvision.transforms.Normalize(normalize_residual_mean, normalize_residual_std)
        residual_norm = normalize_residual_transform(residual)

        self.inverse_normalize_residual = lambda residual_norm: ((residual_norm *
                                                                  normalize_residual_std[:, np.newaxis, np.newaxis]) +
                                                                  normalize_residual_mean[:, np.newaxis, np.newaxis])

        # Save
        self.targets = residual_norm     # targets  = normalized residual
        self.inputs = coarse_norm        # inputs   = normalized coarse

        # Define limits for plotting (plus/minus 3 sigma)
        self.vmin = normalize_rawdata_mean - 3 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 3 * normalize_rawdata_std

        # Additional channels for constant variables
        self.constant_variables = constant_variables
        if constant_variables is not None:
            print("Opening constant variables file (e.g. orography, land-sea mask)")
            # Open file
            ds_const = xr.open_dataset(constant_variables_path,
                                       engine="netcdf4")

            # Get torch tensors and concatenate
            self.const_var = torch.zeros((self.ntime,
                                          len(constant_variables),
                                          self.nx,
                                          self.ny),
                                         dtype=torch.float)

            for i, const_varname in enumerate(constant_variables):
                const_var = ds_const[const_varname]
                # normalize?
                if const_varname != "lsm":
                    print(f"Normalize {const_varname}")
                    mean_var = const_var.mean()
                    std_var = const_var.std() 
                    # print(f"{const_varname} stats -- mean: {mean_var}, std.: {std_var}")
                    const_var = (const_var - mean_var) / std_var
                elif sea_masking:
                    self.mask = torch.from_numpy((const_var < 0.01).to_numpy())
                self.const_var[:,i,:,:] = torch.from_numpy(const_var.to_numpy()).float().expand((self.ntime,-1,-1))

            if sea_masking:
                self.inputs = self.inputs.masked_fill(self.mask, 0)
                self.targets = self.targets.masked_fill(self.mask, 0)

            self.inputs = torch.concatenate((self.inputs, self.const_var), dim=1)

        # Time embeddings
        self.time = ds_c.time.dt        # in datetime format
        self.year = self.time.year
        self.doy = np.array([(np.cos(self.time.dayofyear*2*np.pi/365.)+1)/2, (np.sin(self.time.dayofyear*2*np.pi/365.)+1)/2])

        # Normalize and convert to numpy (load into mem)
        self.year_norm = (self.year.to_numpy() - 1985.)/100
        self.doy_norm = self.doy

        # Torch arrays and float
        self.year_norm = torch.from_numpy(self.year_norm).float()
        self.doy_norm = torch.from_numpy(self.doy_norm).float()

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.inputs.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        return {"inputs": self.inputs[index],
                "targets": self.targets[index],
                "fine": self.fine[index],
                "coarse": self.coarse[index],
                "year": self.year_norm[index],
                "doy": self.doy_norm[:,index]}

    def residual_to_fine_image(self, residual, coarse_image):
        return coarse_image + self.inverse_normalize_residual(residual)

    def plot_fine(self, image_fine, ax, vmin=-2, vmax=2):
        plt.sca(ax)
        ax.coastlines()
        plt.pcolormesh(self.x, self.y, image_fine,
                       vmin=vmin, vmax=vmax, shading='nearest', transform=self.projection)
        
        # Get corners of the domain
        x_min, x_max = self.x.min(), self.x.max()
        y_min, y_max = self.y.min(), self.y.max()
        
        # Transform corners to lon/lat
        corners = ccrs.PlateCarree().transform_points(
            self.projection, 
            np.array([x_min, x_max, x_min, x_max]),
            np.array([y_min, y_min, y_max, y_max])
        )
        
        lon_min, lon_max = corners[:, 0].min(), corners[:, 0].max()
        lat_min, lat_max = corners[:, 1].min(), corners[:, 1].max()
        
        # Add small buffer
        buffer = 0.5  # degrees
        ax.set_extent([lon_min - buffer, lon_max + buffer, 
                    lat_min - buffer, lat_max + buffer], 
                    crs=ccrs.PlateCarree())

    def plot_all_channels(self, X, Y):
        """Plots T for single image (no batch dimension)"""
        fig, axs = plt.subplots(self.n_var, 2, figsize=(8, 2 * self.n_var),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        for i in range(self.n_var):
            self.plot_fine(X[i], axs[i, 0])
            plt.title(self.varnames[i] + " coarse-res")
            self.plot_fine(Y[i], axs[i, 1])
            plt.title(self.varnames[i] + " fine-res")

        plt.tight_layout()
        return fig, axs

    def plot_batch(self, coarse_image, fine_image, fine_image_pred):
        """Plots T for N samples out of batch, separate
        column for coarse, predicted fine and truth fine"""
        N = min(3, coarse_image.shape[0])
        fig, axs = plt.subplots(self.n_var * N, 3, figsize=(8, 2 * N),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        for j in range(N):
            # Plot batch
            for i in range(self.n_var):
                # Plot channel
                self.plot_fine(coarse_image[j, i], axs[(j * self.n_var) + i, 0],
                               vmin=self.vmin[i], vmax=self.vmax[i]) # type: ignore
                self.plot_fine(fine_image_pred[j, i], axs[(j * self.n_var) + i, 1],
                               vmin=self.vmin[i], vmax=self.vmax[i]) # type: ignore
                self.plot_fine(fine_image[j, i], axs[(j * self.n_var) + i, 2],
                               vmin=self.vmin[i], vmax=self.vmax[i]) # type: ignore

        plt.tight_layout()
        return fig, axs
