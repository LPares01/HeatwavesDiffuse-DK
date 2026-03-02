import os 
import xarray as xr
import argparse


# Get arguments from argparser
parser = argparse.ArgumentParser()
parser.add_argument('--year', metavar='year', type=int)
parser.add_argument('--dir', metavar='dir', type=str)
parser.add_argument('--remove_files', dest='remove_files', action='store_true')

## Provide year and month as input to this file using args
args = parser.parse_args()
year = args.year
datadir = args.dir
remove_files = args.remove_files

for dataset in ["cerra", "era5"]:
    print(dataset.upper())
    # Open first month
    month="01"
    filename = f"{dataset}_samples_{year}{month}.nc4c"
    path_to_file = f"{datadir}{filename}"
    ds = xr.open_dataset(path_to_file, engine="netcdf4")

    for m in range(2,13):
        month = f"{m:02d}"
        filename = f"{dataset}_samples_{year}{month}.nc4c"
        path_to_file = f"{datadir}{filename}"
        ds2 = xr.open_dataset(path_to_file, engine="netcdf4")

        # Concatenate along time axis
        ds = xr.concat((ds, ds2), dim="time")
        print("Concat'd month", m)
        
        ds2.close()

    # Save
    save_file = f"{dataset}_samples_{year}.nc4c"
    ds.to_netcdf(f"{datadir}{save_file}")

    ds.close()

    if remove_files:
        print("Removing intermediate files")
        for m in range(1,13):
            month = f"{m:02d}"
            filename = f"{dataset}_samples_{year}{month}.nc4c"
            path_to_file = f"{datadir}{filename}"
            os.remove(f"{datadir}{filename}")

