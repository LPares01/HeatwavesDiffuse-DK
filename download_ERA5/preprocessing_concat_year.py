import os 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import argparse
import psutil

# Get arguments from argparser
parser = argparse.ArgumentParser()
parser.add_argument('--year', metavar='year', type=int)
parser.add_argument('--data', metavar='data', type=str)
parser.add_argument('--remove_files', metavar='remove_files', action=argparse.BooleanOptionalAction)

## Provide year and month as input to this file using args
args = parser.parse_args()
year = args.year
datadir = args.data
remove_files = args.remove_files

# Open first month
month="01"
filename = f"samples_{year}{month}.nc"
path_to_file = f"{datadir}{filename}"
ds = xr.open_dataset(path_to_file, engine="netcdf4")
print("l27:", psutil.virtual_memory())


for m in range(2,13):
    month = f"{m:02d}"
    filename = f"samples_{year}{month}.nc"
    path_to_file = f"{datadir}{filename}"
    ds2 = xr.open_dataset(path_to_file, engine="netcdf4")
    print("l35:", psutil.virtual_memory(), "\n m =", m)

    # Concatenate along time axis
    ds = xr.concat((ds, ds2), dim="time")
    print("Concat'd month", m)
    
    ds2.close()
    print("l42:", psutil.virtual_memory(), "\n m =", m)

# Save
save_file = f"samples_{year}.nc"
ds.to_netcdf(f"{datadir}{save_file}")

ds.close()

if remove_files:
    print("Removing intermediate files")
    for m in range(1,13):
        month = f"{m:02d}"
        filename = f"samples_{year}{month}.nc"
        path_to_file = f"{datadir}{filename}"
        os.remove(f"{datadir}{filename}")

