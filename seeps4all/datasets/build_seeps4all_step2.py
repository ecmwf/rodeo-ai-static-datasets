#
# Build a SEEPS4ALL dataset 
# (requires to run build_seeps4ll_step1.ipy first)
#
#

import copy
import numpy as np
import pandas as pd
import xarray as xr

#location identifier
ll = lambda icnt: int(lats[icnt]*10000)*1000+lons[icnt]

# Settings
# weather parameter
param = "tp24"
# define period of interest 
year_i = 2022
year_f = 2024

# Climatology years
year_clim_i = 1991
year_clim_f = 2020

# Raw ECAD directory (get data from https://www.ecad.eu)
path_input="/my/input/data/path/"

# output directory (same output directory as in build_seeps4all_step1.py)
path_output="/my/ouput/data/path/"

# Get station list and metadata 

# read file with general info abou the data 
clim_years = f"{year_clim_i}-{year_clim_f}"
clim_st = pd.read_csv(f"{path_output}/local_climate_{clim_years}/stations_info_{clim_years}.csv",index_col=0)

# metadata
lats= clim_st.LAT.values
lons= clim_st.LON.values
elevs = clim_st.ELEV.values
stnid_list = clim_st.STNID.values

del clim_st

#Loop over stations and dates to build the dataset

# QC1 parameter
p99_0 = 200

# one season + to account for max forecat step (here 15 days)
time_start = pd.Timestamp("%s0101"%year_i)
time_end   = pd.Timestamp("%s1231"%year_f)+pd.Timedelta(value=15,unit="D")
ndates = len(pd.date_range(time_start,time_end,freq="1D"))

print(f"Prepare OBS files from {year_i} to {year_f}")

# initialise main outpout
oclim_all = xr.Dataset()
seeps_all = xr.Dataset()

# description and licence
description_seeps="observations with SEEPS thresholds and coefficients"
description_clim="observations with climate percentiles from 1 to 99 and maximum"
licence="CC-BY-NC. See also https://knmi-ecad-assets-prd.s3.amazonaws.com/documents/ECAD_datapolicy.pdf"
version="1.1.0"

#initialisation
latlons = []

# loop over station ids
for icnt, stnid in enumerate(stnid_list):
    print("")
    print(f"{icnt+1}/{len(stnid_list)}, ... , station {stnid}")

    # read raw ECAD data for one station
    filein = "%s/RR_STAID0%05d.txt"%(path_input,stnid) 
    data = pd.read_csv(filein, sep=",", header=22, names=("STNID","SOUID","DATE","RR","Q_RR") )

    # percipitation and quality control values
    val =  np.array([ float(v)/10. for v in data["RR"] ])
    qc  =  np.array([ int(v) for v in data["Q_RR"] ])
    # remove poor quality observations
    val [ qc > 0 ] = np.nan
    del qc
    
    # apply QC 1
    idno = ( val > 5*p99_0*np.cos(lats[icnt]*np.pi/180) )
    val [ idno ] = np.nan
    # apply QC 2
    uni = np.unique(val)
    for bigv in uni[uni>120]:
        if (val == bigv).sum() >1:
            val[ val == bigv ] = np.nan

    # apply QC 3
    # proportion of zeros
    prop_zeros = (val<0.1).sum()/(val>=0).sum()
    if prop_zeros > 0.99:
        print(f"{stnid} rejected because proportion of zeros= {np.round(prop_zeros,2)}")  
        continue

    # apply QC 4
    # check for multiple stations at the same location 
    if ll(icnt) in latlons:
        continue
    else:
        latlons.append(ll(icnt))

    # time with frame     
    # +24h: end of accumulation time (rather than begining) 
    tim =  np.array([ pd.Timestamp(str(v))+pd.Timedelta(24,unit="h") for v in data["DATE"] ] )
    del data 
 
    # define periof of interest  
    dok = (tim >= time_start) & (tim <= time_end)
    # check availability
    avail = np.nanmean(np.isfinite(val[dok]))
    if avail == 0:
        print("not data available for this station")
        continue
    if avail < 0.:
        print("availability:",avail, ": too low!")
        continue

    # select data of interest
    nx = dok.sum()
    x = np.array(val[dok]).reshape((nx,1))
    x [ np.isfinite(x) == False ] =np.nan

    # PART 1: SEEPS coefficients
    # --------------------------
    
    # info to derive equitable coefficients for SEEPS
    clim_eq = pd.read_csv(f"{path_output}/local_climate_{clim_years}/equitable_categories_st{stnid}_{clim_years}.csv")
    if len(clim_eq)<12:
        print("availability of climatology not for the 12 month of the year")
        continue

    # set to nan when needed
    p1,p3 = clim_eq["p1"].values,clim_eq["p2"].values
    idna = (p1 < 0.1) | (p1 >= 0.85) | (clim_eq["count_all"] < 300) | (clim_eq["count_t1"] < 50)
    p1[idna],p3[idna] = np.nan, np.nan

    # initialise SEEPS matrix 
    seeps_coefs = np.zeros((12,13)) # 12 months, 13 coefficients

    # compute SEEPS coefficients
    seeps_coefs[:,0] = 1 - clim_eq["p1"]
    seeps_coefs[:,1] = clim_eq["t1"]
    seeps_coefs[:,2] = 1 - clim_eq["p2"]
    seeps_coefs[:,3] = clim_eq["t2"]
    
    val = 0.5/(1-p1)
    val[val>12.5] = 12.5
    seeps_coefs[:,5] = val
    
    val = 0.5/(1-p3) + 0.5/(1-p1)
    val[val>50] = 50
    seeps_coefs[:,6] = val
    
    val = 0.5/p1
    val[val>12.5] = 12.5
    seeps_coefs[:,7] = val
    
    val = 0.5/(1-p3)
    val[val>37.5] = 37.5
    seeps_coefs[:,9] = val
    
    val = 0.5/p1 + 0.5/p3
    val[val>13.235] = 13.235
    seeps_coefs[:,10] = val

    val = 0.5/p3
    val[val>1] = 1
    seeps_coefs[:,11] = val
    
    seeps_labs =("p1","t1","p2","t2")
    for im in range(9):
        seeps_labs += ("m%s"%(im+1),)

    coords_seeps=dict( time = tim[dok],
                 stnid = [stnid,],
                 name = list(seeps_labs),
                 month = list(range(1,12+1)),
                 lat= ("stnid", [np.round(lats[icnt],2),]),
                 lon= ("stnid", [np.round(lons[icnt],2),]),
                 elevation= ("stnid", [np.round(elevs[icnt],2),]),
                )
    
    # ouput dataset for one stations
    data_vars=dict()
    data_vars["observation"] = ([ "time","stnid"], x)
    data_vars["coefficients"] = ([ "month","stnid","name"], seeps_coefs.reshape((12,1,13)))

    # concatenate over stations
    if icnt == 0:
        attrs=dict(description=description_seeps,licence=licence,version=version)
        seeps_all = xr.Dataset(data_vars=data_vars,coords=coords_seeps,attrs=attrs)
    else:   
        seeps = xr.Dataset(data_vars=data_vars,coords=coords_seeps)
        seeps_all = xr.concat([seeps_all,seeps],dim="stnid")

    # PART 2: climatology percentiles
    # -------------------------------

    # read info about percentiles of the climate distribution
    clim_perc = pd.read_csv(f"{path_output}/local_climate_{clim_years}/percentiles_st{stnid}_{clim_years}.csv")

    coords_perc=dict( time = tim[dok],
                      stnid = [stnid,],
                      perc = list(range(1,100+1)),
                      month = list(range(1,12+1)),
                      lat= ("stnid", [np.round(lats[icnt],2),]),
                      lon= ("stnid", [np.round(lons[icnt],2),]),
                      elevation= ("stnid", [np.round(elevs[icnt],2),]),
                    )
    
    data_vars=dict()
    data_vars["observation"] = ([ "time","stnid"], x)
    data_vars["percentile"] = ([ "month","stnid","perc"], clim_perc.values[:,2:].reshape((12,1,99+1)))

    # concatenate over stations
    if icnt == 0:
        attrs=dict(description=description_clim,licence=licence,version=version),
        oclim_all = xr.Dataset(data_vars=data_vars,coords=coords_perc,attrs=attrs)
    else:    
        oclim = xr.Dataset(data_vars=data_vars,coords=coords_perc)
        oclim_all = xr.concat([oclim_all,oclim],dim="stnid")

# Archive in NetCDF

# define name of output file with SEEPS coeffs
file_seeps_zarr = f"{path_output}/obs_seeps_{year_i}_{year_f}_ecad.zarr"
print("save as ZARR")
print("... ", file_seeps_zarr)
seeps_all.to_zarr(file_seeps_zarr)

# define name of output file for climatology percentiles 
file_oclim_zarr = f"{path_output}/obs_clim_{param}_{year_i}_{year_f}_ecad.zarr"
print("save as ZARR")
print("... ",file_oclim_zarr)
oclim_all.to_zarr(file_oclim_zarr)

#the end






