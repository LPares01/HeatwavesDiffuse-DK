import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load CERRA data
ds = xr.open_dataset('/a/cerra/nc/file', engine='netcdf4')

# Select a specific time step
t2m = ds['2t'].isel(time=0, height=0)

# Convert from Kelvin to Celsius
t2m_celsius = t2m - 273.15

# Create figure with map projection
lambert_proj = ccrs.LambertConformal(
    central_longitude=8, 
    central_latitude=50, 
    standard_parallels=(50, 50)
)
lambert_proj.threshold /= 1000

fig = plt.figure()
ax = plt.axes(projection=lambert_proj)

# Determine symmetric range around 0°C for colormap
vmax = max(abs(t2m_celsius.min().values), abs(t2m_celsius.max().values))
vmin = -vmax

# Plot the data
im = ax.contourf(t2m_celsius.x - t2m_celsius.x.max()/2, 
                  t2m_celsius.y - t2m_celsius.y.max()/2, 
                  t2m_celsius,
                  levels=20,
                  cmap='RdBu_r',
                  vmin=vmin,
                  vmax=vmax,
                  transform=lambert_proj)

# Add map features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.8)
cbar.set_label('2-meter Temperature (°C)', fontsize=12)

# Add title
plt.title(f'CERRA 2-meter Temperature\n{t2m.time.values}', fontsize=14)

plt.tight_layout()
plt.show()
