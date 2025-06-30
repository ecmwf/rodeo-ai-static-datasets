
"""Data routines for SEEPS4ALL.

Open data from directory or web.
Select domain.
"""

import os
import numpy as np
import xarray as xr

url = "https://object-store.os-api.cci2.ecmwf.int"
bucket = "ecmwf-rodeo-benchmark"

# open observation file 
def get_obs(path_data,obs_type,obs_origin="ecad",years="2022_2024",param="tp24"):
    
    path_zarr = f"{path_data}/obs_{obs_type}_{param}_{years}_{obs_origin}.zarr"
    if  os.path.isdir(path_zarr) == False:
        if not os.path.exists(path_data):
            print(f". creating {path_data}")
            os.makedirs(path_data)
        print(f". fetching data from {url}")
        bucket = "ecmwf-rodeo-benchmark"
        ds = xr.open_zarr(f'{url}/{bucket}/seeps4all/obs_{obs_type}_{param}_{years}_{obs_origin}.zarr',decode_timedelta=True)
        ds.to_zarr(path_zarr)
    print(f"open: {path_zarr}")
    obs_seeps_data = xr.open_zarr(path_zarr,decode_timedelta=True)
    
    nobs  = len(obs_seeps_data["stnid"])
    print("... total number of observation locations:", nobs)
    
    return obs_seeps_data
    
# open forecast files 
def get_fct(path_data,name_forecasts):
    
    nexp = len(name_forecasts)
    fct_data = []
    for iex in range(nexp):
        path_zarr=f"{path_data}/{name_forecasts[iex]}.zarr"
        if  os.path.isdir(path_zarr) == False:
            print(f". get data from {url}")
            ds = xr.open_zarr(f'{url}/{bucket}/seeps4all/{name_forecasts[iex]}.zarr',decode_timedelta=True)
            ds.to_zarr(path_zarr)
        else:    
            print(f"open: {path_zarr}")
            data = xr.open_zarr(path_zarr,decode_timedelta=True)
            fct_data.append(data)
            
            runs = data.run.values
            nruns = len(runs)
            steps  = data.step.values
            nsteps = len(steps)
            
            print("Number of forecast steps:",nsteps)
            print("Number of forecast runs:",nruns)
            
    return fct_data

def info_domain(domain:str,
                lats:np.ndarray,
                lons:np.ndarray) -> np.ndarray:
    """ Domain of inerest for score computation """
    if domain == "europe":
        dom_lat=[35,71]
        dom_lon=[-10,35]
    elif domain == "tropics":
        dom_lat=[-30,30]
        dom_lon=[-180,180]
    else:
        raise NameError("Please define domain:", domain)
    
    dom_id = (lats > dom_lat[0]) & (lats <=dom_lat[1]) & (lons > dom_lon[0]) & (lons <=dom_lon[1])
    
    return dom_id

# select domain
def select_domain(data:xr.Dataset, 
                  domain:str)-> xr.Dataset:
    """ select domain of interest. """

    lats = data.lat.values.flatten()
    lons = data.lon.values.flatten()
    
    dom_id = info_domain(domain,lats,lons)
    data = data.sel(stnid = data.stnid.values.flatten()[dom_id])

    return data

def get_domain(obs_data:xr.Dataset, 
               fct_data:xr.Dataset, 
               domain:str):

    print(f"focus on domain {domain}") 
    obs_data = select_domain(obs_data,domain)
    for iex in range(len(fct_data)):
        fct_data[iex] = select_domain(fct_data[iex],domain)

    return obs_data,fct_data
    
