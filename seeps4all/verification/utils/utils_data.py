
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
        ds = xr.open_zarr(f'{url}/{bucket}/seeps4all/obs_{obs_type}_{param}_{years}_{obs_origin}.zarr')
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
            ds = xr.open_zarr(f'{url}/{bucket}/seeps4all/{name_forecasts[iex]}.zarr')
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


# select obs according to fct
def select_obs (obs_data,fct_data,obs_type,offset = 0):

    nexp = len(fct_data)
    valid_t_all = []
    for iex in range(nexp):
        runs = fct_data[iex].run.values
        steps = fct_data[iex].step.values
        for istep,step in enumerate(steps):
            # validity date
            vtime = [pd.Timestamp(vt)-pd.Timedelta(offset,unit="h") for vt in runs+step ]
            valid_t_all += ["%s-%02d-%02d"%(vt.year,vt.month,vt.day) for vt in vtime]
    # select all observations
    time_obs = np.unique(valid_t_all)
    data = obs_data.sel(time=time_obs)
    
    ndates = len(time_obs)
    nstid = len(data.stnid)
    
    if obs_type == "seeps":
        seeps_mtx = np.zeros((ndates,nstid,13))
    elif obs_type == "clim":
        perc_mtx = np.zeros((ndates,nstid,99+1))

    for ida, dd in enumerate(time_obs):
        date = pd.to_datetime(dd)
        datei = "%s%02d%02d"%(date.year,date.month,date.day)
        montho = pd.Timestamp(datei).month
        if obs_type == "clim":
            perc_mtx[ida,:,:] = data.percentile.sel(month=montho).values
        elif obs_type == "seeps":
            seeps_mtx[ida,:,:] = data.coefficients.sel(month=montho).values
        
    if obs_type == "clim":
        for ik in range(100):
            data = data.assign(perc=(["time","stnid"],perc_mtx[:,:,ik]))
            data = data.rename(perc="perc%s"%(1+ik))
            
    elif obs_type == "seeps":
        data = data.assign(p1=(["time","stnid"],seeps_mtx[:,:,0]))
        data = data.assign(t1=(["time","stnid"],seeps_mtx[:,:,1]))
        data = data.assign(p2=(["time","stnid"],seeps_mtx[:,:,2]))
        data = data.assign(t2=(["time","stnid"],seeps_mtx[:,:,3]))
        for ik in range(9):
            data = data.assign(mx=(["time","stnid"],seeps_mtx[:,:,4+ik]))
            data = data.rename(mx="mx%s"%(1+ik))

    return data
    
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
    
