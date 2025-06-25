#
# Get SINGLE FORECASTS aligned with ECAD data
#
#  input: forecast data as a grib file
#      +  info about station locations
#  ouput: collocated data as a netcdf file
#

import copy
import numpy as np
import pandas as pd
import metview as mv
import xarray as xr

# SETTINGS
param = "tp24"
obs_origin = "ecad"
years_clim = "2022_2024"

path_data="/path/to/my/obs/data"
path_data_grib="path/to/my/raw/forecast"

#INPUT FILE: forecast grib file 
#
file_name= "ifs_tp_20240601_to_20240831_00"

path_grib=f"{path_data_grib}/{file_name}.grib"
print("open grib:",path_grib)
fcst = mv.read(path_grib)

fcst_dataset = fcst.to_dataset()   
dates = fcst_dataset.time.values
steps = fcst_dataset.step.values
if steps[0] == 0:
    steps = steps[1:]

#INPUT FILE: info climato 
path_zarr=f"{path_data}/obs_seeps_{param}_{years_clim}_{obs_origin}.zarr"
print("open zarr:",path_zarr)
obs = xr.open_zarr(path_zarr)

# derive primary info
stnid = obs.stnid.values.flatten()
lats = obs.lat.values.flatten()
lons = obs.lon.values.flatten()

# metadata
list_steps = [ pd.Timedelta(step.astype('timedelta64[h]'),unit="H") for step in steps ] 
list_dates = [ pd.Timestamp(time) for time in dates ] 

# initialisation
fct_all = np.zeros([len(steps),len(dates),len(lats)]) * np.nan

# loop over forecat lead times
for idd,fdate in enumerate(list_dates):
    print("forecast date:",fdate)
    datein = fdate.year*10000+fdate.month*100+fdate.day
    fcv_p = 0.
    for itt,step in enumerate(steps):
        stepin = step.astype('timedelta64[h]').astype(int)
        fcv = mv.nearest_gridpoint(fcst.select(step=stepin,date=datein),np.array(lats),np.array(lons),"valid")      
        fct_all[itt,idd,:] = (fcv-fcv_p)*1000
        fcv_p = copy.copy(fcv)

# ouput dataset
out = xr.Dataset(
        data_vars=dict(
           forecast = ([ "step","run","stnid"], fct_all),
        ),
        coords=dict( stnid = stnid,
                     run   = list_dates,  
                     step  = list_steps, 
                     lat= ("stnid", lats),
                     lon= ("stnid", lons),
        ),
        attrs=dict(description="forecasts at station locations"),
    )

# archive
file_fct_zarr = f"{path_data}/fct_{file_name}_{obs_origin}.zarr"
print("save as ZARR")
print("... ", file_fct_zarr)
out.to_zarr(file_fct_zarr)

# the end

