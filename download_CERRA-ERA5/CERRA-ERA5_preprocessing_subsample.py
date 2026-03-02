import numpy as np
import xarray as xr
import random
import argparse

# Get arguments from argparser
parser = argparse.ArgumentParser()
## Arguments
parser.add_argument('--year', metavar='year', type=int)
parser.add_argument('--month', metavar='month', type=int)
parser.add_argument('--last_day', metavar='last_day', type=int)
parser.add_argument('--dir', metavar='dir', type=str)
parser.add_argument('--dummy', dest='dummy', action='store_true')

## Provide year and month as input to this file using args
args = parser.parse_args()
year = args.year
month = args.month
last_day = args.last_day
datadir = args.dir
dummy = args.dummy

# Set random seed for reproducibility, but different for each year/month
seed = year*12 + month
print(seed)
random.seed(seed)

 # Select time inds randomly 
time_inds = np.arange(last_day, dtype=int)
random.shuffle(time_inds) # type: ignore

for dataset in ["cerra", "era5"]:
        print(dataset.upper())
        ## First variable for setting up
        filename = f"{dataset.upper()}_max/{year}/max/{dataset}_t2m_max_{year}{month:02d}.nc4c"

        # Open file
        path_to_file = f"{datadir}{filename}"
        ds = xr.open_dataset(path_to_file, engine="netcdf4")

        # Pre-processed dataset
        if not dummy:
                ds_proc = ds.isel(time=time_inds)
                save_file = f"{dataset}_samples_{year}{month:02d}.nc4c"
        else:
                ds_proc = ds.isel(time=time_inds[0])
                save_file = f"dummy/{dataset}_samples_{year}{month:02d}.nc4c"

        ds_proc.to_netcdf(f"{datadir}{save_file}")
