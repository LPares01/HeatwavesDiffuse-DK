import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load ERA5 data
ds = xr.open_dataset('/a/era5/nc/file', engine='netcdf4')

# Select a specific time step
t2m = ds['2t'].isel(time=0)

# Convert from Kelvin to Celsius
t2m_celsius = t2m - 273.15

# Create figure with map projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Determine symmetric range around 0°C for colormap
vmax = max(abs(t2m_celsius.min().values), abs(t2m_celsius.max().values))
vmin = -vmax

# Plot the data
im = ax.contourf(t2m_celsius.lon, 
                  t2m_celsius.lat, 
                  t2m_celsius,
                  levels=20,
                  cmap='RdBu_r',
                  vmin=vmin,
                  vmax=vmax,
                  transform=ccrs.PlateCarree())

# Add map features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('2-meter Temperature (°C)', fontsize=12)

# Add title
plt.title(f'ERA5 2-meter Temperature\n{t2m.time.values}', fontsize=14)

plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('era5_t2m_map.png', dpi=300, bbox_inches='tight')
