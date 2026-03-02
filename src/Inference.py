import torch
from Network import UNet, EDMPrecond
import numpy as np
from DatasetUS import UpscaleDatasetCERRA
import os
import netCDF4 as nc
from datetime import datetime
import xarray as xr
import pandas as pd

@torch.no_grad()
def sample_unet(input_batch, model, device, dataset):

    images_input = input_batch["inputs"].to(device)
    coarse = input_batch["coarse"]
    condition_params = torch.cat(
                (input_batch["doy"].to(device),
                 input_batch["year"].unsqueeze(1).to(device)), dim=1)
    residual = model(images_input, class_labels=condition_params)
    predicted = dataset.residual_to_fine_image(residual.detach().cpu(), coarse)

    return predicted


@torch.no_grad()
def sample_model_EDS(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1):

    images_input = input_batch["inputs"].to(device)
    coarse, fine = input_batch["coarse"], input_batch["fine"]

    condition_params = (input_batch["doy"].to(device)).unsqueeze(1)
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 3, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))

        # Euler step.
        denoised = model(x_hat, t_hat, images_input, condition_params).to(
            torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input,
                             condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse)


    return coarse, fine, predicted

def save_predictions_to_netcdf(predictions, dates, output_path):
    """
    Save predictions to a NetCDF4 file.
    """

    # Create NetCDF file with NETCDF4_CLASSIC format for better compatibility
    with nc.Dataset(output_path, 'w', format='NETCDF4_CLASSIC') as ncfile:
        # Create dimensions
        n_samples, n_channels, height, width = predictions.shape
        ncfile.createDimension('time', n_samples)
        ncfile.createDimension('channel', n_channels)
        ncfile.createDimension('height', height)
        ncfile.createDimension('width', width)
        
        # Create prediction variable
        pred_var = ncfile.createVariable('predictions', 'f4', 
                                         ('time', 'channel', 'height', 'width'),
                                         zlib=True, complevel=4)
        # Add variable attributes
        pred_var.long_name = 'Model predictions'
        pred_var.units = 'K'
        
        # Add date
        time_var = ncfile.createVariable('time', 'i4', ('time',))
        time_var.long_name = 'time'
        time_var.units='hours since 1985-01-01'
        time_var[:] = nc.date2num(dates, time_var.units, calendar='proleptic_gregorian')
        
        # Write predictions
        pred_var[:] = predictions
        
        # Add global attributes
        ncfile.description = 'Model predictions'
        ncfile.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ncfile.source = 'Inference.py'
        ncfile.conventions = 'CF-1.6'
    
    print(f"Saved predictions to {output_path}")


@torch.no_grad()
def run_full_inference(model_type, model, year, datadir, dataloader, dataset, output_dir, device):
    """
    Run inference on entire test set and save predictions.
    """

    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = []
    
    print(f"Running inference for {year}...")
    
    for batch in dataloader:
        if model_type == 'unet':
            predicted = sample_unet(batch, model, device, dataset)
        elif model_type == 'diffusion':
            _, _, predicted = sample_model_EDS(batch, model, device, dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store prediction
        predictions.append(predicted.numpy())
        
    
    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)

    # Get prediction dates
    dates = xr.open_dataset(datadir + f"cerra_samples_{year}.nc4c", engine='netcdf4').time.values[:len(predictions)]

    # Sort predictions by date
    dates = np.array(dates)
    sort_idx = np.argsort(dates)
    dates = pd.to_datetime(dates[sort_idx]).to_pydatetime()
    predictions = predictions[sort_idx, :, :, :]
    
    print(f"Total samples processed: {predictions.shape[0]}")

    output_path = output_dir + f"{model_type}_{year}.nc4c"

    save_predictions_to_netcdf(predictions, dates, output_path)

    return predictions

if __name__ == "__main__":

    run_nbrs = ['04']
    sea_masking = [False]

    for i in range(len(run_nbrs)):
        # configuration
        run_nbr = run_nbrs[i]
        model_type = 'unet' # 'unet' or 'diffuse'
        dataset = 'full'
        batch_size = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        output_dir = os.path.join(os.path.dirname(__file__), 
                                f"./runs_{model_type}/run_{run_nbr}/inference_results/")

        if model_type == 'unet':
            model = UNet((256, 256), 3, 1, label_dim=3, use_diffuse=False).to(device)
        else:
            model = EDMPrecond((256, 256), 4, 1, label_dim=3).to(device)

        model_path = os.path.join(os.path.dirname(__file__), 
                                f"./runs_{model_type}/run_{run_nbr}/run_{run_nbr}.pt")

        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")

        if dataset == 'dummy':
            datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/dummy/")
        else:
            datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/test_samples/")
        
        if dataset == 'full':
            test_start, test_end = 2015, 2020
        else:
            test_start, test_end = 1990, 1990

        for year in range(test_start, test_end+1):

            dataset_test = UpscaleDatasetCERRA(datadir, year_start=year, year_end=year,
                                        constant_variables=["lsm", "orog"], sea_masking=sea_masking[i])
            
            dataloader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
            
            predictions = run_full_inference(
                model_type,
                model, 
                year,
                datadir,
                dataloader_test,
                dataset_test,
                output_dir,
                device
            )
        
        # visualize_samples(predictions, targets, inputs, output_dir, num_samples=20)
        
        print("\nInference complete!")
        print(f"Results saved to: {output_dir}")
        

        # # Try model
        # if model_type == 'unet':
        #     coarse, fine, predicted = sample_unet(dataset_test[0:4], model,
        #                                             device, dataset_test)
            
        #     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #     ax[0].pcolormesh(coarse[0, 0])
        #     ax[0].set_title("Coarse")
        #     ax[1].pcolormesh(fine[0, 0])
        #     ax[1].set_title("Fine")
        #     ax[2].pcolormesh(predicted[0, 0])
        #     ax[2].set_title("Predicted")

        #     plt.show()
        # else:
        #     coarse, fine, predicted = sample_model_EDS(dataset_test[0:4], model,
        #                                             device, dataset_test)
        #     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #     ax[0].pcolormesh(coarse[0, 0])
        #     ax[0].set_title("Coarse")
        #     ax[1].pcolormesh(fine[0, 0])
        #     ax[1].set_title("Fine")
        #     ax[2].pcolormesh(predicted[0, 0])
        #     ax[2].set_title("Predicted")
