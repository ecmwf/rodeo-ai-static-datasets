"""Statistics related routines for SEEPS4ALL.

Bootstrapping, weighting, and dressing functions.
"""

import copy
import numpy as np
import xarray as xr
from utils.utils_scores import fbi_ct
from utils.utils_scores import ets_ct
from utils.utils_scores import hss_ct
from utils.utils_scores import pss_ct
from utils.utils_scores import ora_ct

def param_bootstrap()->tuple[int,int]:
    """ Bootstrap parameter."""
    nboot  = 5000
    nblock = 5
    return nboot,nblock

def angular_distance_haversine(lat1:np.ndarray,lon1:np.ndarray,
                               lat2:np.ndarray,lon2:np.ndarray)->np.ndarray:
    sdlat = np.sin(np.radians((lat2-lat1)*0.5))
    sdlon = np.sin(np.radians((lon2-lon1)*0.5))
    return 2.*np.arcsin(np.sqrt(sdlat*sdlat+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*sdlon*sdlon))
#
# 
#
def station_weighting(data:xr.Dataset,
                      scangle:float=0.75)-> np.ndarray:
    """ Compute weighting as a function of station denstity."""
    
    lats = data.lat.values.flatten()
    lons = data.lon.values.flatten()
     
    scangle*=np.pi/180.
    weights = []
    for lat,lon in zip(lats,lons):
        distances = angular_distance_haversine(lats,lons,lat,lon)
        kernel = np.exp(-(distances/scangle)**2)
        kernel[distances>4.*scangle] = 0.
        w = 1./kernel.sum()
        weights.append(w)  
     
    return np.array(weights) 

def get_weights(data:xr.Dataset,wtype:str)-> np.ndarray:
    if wtype == "uniform":
        print("unifrom weights")
        weights = np.ones(len(data.lat.values.flatten()))
    elif  wtype == "station_weighting":
        print("weights based on station density")
        weights = station_weighting(data) 
    else:
        raise NameError(f"Please define weighting type {wtype}")
    return weights

def sigma_precip(deltax:float)->list:
    """ Accounting for observation uncertainty 
    following Ben Bouallegue et al 2020.
    Sigma for TP. """
    
    alpha0 = 0.005*deltax**0.5
    alpha1 = 1.
    beta0  = 0.0005*deltax
    beta1  = -0.02* deltax + 0.55 *deltax**0.5
    delta  = 0.005*deltax**0.5
    sigma  = [alpha0, alpha1, beta0, beta1, delta]
    return sigma

def shifted_gamma_det(f:np.ndarray,
                      sigma:list)-> np.ndarray:
    """ Dress single forecast."""
    f0 = copy.deepcopy(f)
    f0 [ np.where( f0 < 0. ) ]= 0.

    mu     = sigma[0] + sigma[1] * f0
    sig_sq = sigma[2] + sigma[3] * f0**0.5

    shape  = (mu**2)/(sig_sq**2)
    scale  = sig_sq**2/mu

    shape [np.where( sig_sq == 0. )]= 0.
    scale [np.where( mu     == 0. )]= 0.

    rdField = np.random.gamma(shape,scale) - sigma[4]
    rdField [np.where( rdField < 0 )]=0.

    return rdField

def shifted_gamma_ens(f:np.ndarray,sigma:list,nmem=50):
    """ Perturb ensemble forecast."""
    
    ens = np.zeros((nmem,len(f)))

    f0 = copy.deepcopy(f)
    f0 [ np.where( f0 < 0. ) ]= 0.

    mu     = sigma[0] + sigma[1] * f0
    sig_sq = sigma[2] + sigma[3] * f0**0.5

    shape  = (mu**2)/(sig_sq**2)
    scale  = sig_sq**2/mu

    shape [np.where( sig_sq == 0. )]= 0.
    scale [np.where( mu     == 0. )]= 0.

    for imem in range(nmem):
        rdField = np.random.gamma(shape,scale) - sigma[4]
        rdField [np.where( rdField < 0 )]=0.
        ens[imem,:] = rdField
        
    return ens

def dressing(fct_data:xr.Dataset,
             delta_x:float,
             nmem:int=50)->tuple:
    """ Dress single forecast."""
    
    ens_data = []
    for iex in range(len(fct_data)):
        # input data
        fct = fct_data[iex]["forecast"].values

        # uncertainty coeeficient
        sigma = sigma_precip(delta_x[iex])

        # draw from shifted gamma distributions
        fct_dressed = shifted_gamma_ens(fct.flatten(),sigma,nmem=nmem)

        # Ensemble data
        # -------------
        # transform to the expected shape
        fct_dressed = fct_dressed.reshape( (nmem,)+fct.shape )
        # define xarray
        ens = xr.Dataset(
            data_vars=dict(
               forecast = ([ "number","step","run","stnid"], fct_dressed),
            ),
            coords=dict( stnid  = fct_data[iex]["stnid"].values,
                     run    = fct_data[iex]["run"].values,  
                     step   = fct_data[iex]["step"].values,
                     number = range(nmem) 
            ),
            attrs=dict(description="values"),
        )
        # with metadata
        ens = ens.assign_coords(lon=("stnid",fct_data[iex]["lon"].values))
        ens = ens.assign_coords(lat=("stnid",fct_data[iex]["lat"].values))
 
        ens_data.append(ens)
    return ens_data
    
def get_boot_ids(ndates= 1000,nblock= 5)-> np.ndarray:
    """ Block-bootstrapping slection."""
    i0 = np.random.choice(range(ndates%nblock+1))
    id_block = np.random.choice(range(int(ndates/nblock)),int(ndates/nblock))
    id_boot = [ list(range(iday*nblock+i0,(iday+1)*nblock+i0))for iday in id_block  ] 
    id_boot = np.array(id_boot).flatten()
    return id_boot

def simple_bootstrap(sco:np.ndarray,
                     bootstrap:bool)->np.ndarray:
    """ Apply block-boostrapping when no keys."""
    
    if bootstrap:
        nboot,nblock = param_bootstrap()
    else:   
        nboot = 1
        
    nexp   = sco.shape[1]
    nsteps = sco.shape[2]
    ndates = sco.shape[3]
        
    sc_dist =np.zeros( (nboot,2,nexp,nsteps) )
    for ib in range(nboot):
        if nboot >1:
            id_boot = get_boot_ids(ndates,nblock)
        else:
            id_boot = range(ndates)
        for iex in range(nexp):
            for ir in range(2):
                sc_dist[ib,ir,iex,:] = np.mean(sco[ir,iex,:,id_boot],axis=0)    
                
    return sc_dist

def sc_bootstrap(sco:np.ndarray,
                 bootstrap:bool)->np.ndarray:
    """ Apply block-boostrapping when keys available ."""

    if bootstrap:
        nboot,nblock = param_bootstrap()
    else:   
        nboot = 1
        
    nexp   = sco.shape[1]
    nkeys  = sco.shape[2]
    nsteps = sco.shape[3]
    ndates = sco.shape[4]
    
    sc_dist =np.zeros( (nboot,2,nexp,nkeys,nsteps) )

    for ib in range(nboot):
        if nboot >1:
            id_boot = get_boot_ids(ndates,nblock)
        else:
            id_boot = range(ndates)   
        for iex in range(nexp):
            for ik in range(nkeys):
                for ir in range(2):
                    sc_dist[ib,ir,iex,ik,:] = np.mean(sco[ir,iex,ik,:,id_boot],axis=0)
    
    return sc_dist

def ct_bootstrap (CT:np.ndarray,
                  score_name:str,
                  bootstrap:bool)->np.ndarray:
    """ Apply block-boostrapping to CT-based scores."""

    if bootstrap:
        nboot,nblock = param_bootstrap()
    else:   
        nboot = 1

    if score_name[0:3] == "FBI":
        score_ct = fbi_ct
    elif score_name[0:3] == "ETS":
        score_ct = ets_ct
    elif score_name[0:3] == "PSS":
        score_ct = pss_ct
    elif score_name[0:3] == "ORA":
        score_ct = ora_ct
    elif score_name[0:3] == "HSS":
        score_ct = hss_ct
    else:
        raise NameError(f"Please define score {score_name[0:3]}")
    
    nexp   = CT["A"].shape[1]
    nkeys  = CT["A"].shape[2]
    nsteps = CT["A"].shape[3]
    ndates = CT["A"].shape[4]
    
    sc_dist =np.zeros( (nboot,2,nexp,nkeys,nsteps) )
    for ib in range(nboot):
        if nboot >1:
            id_boot = get_boot_ids(ndates,nblock)
        else:
            id_boot = range(ndates)
        for iex in range(nexp):
            for ik in range(nkeys):
                for ir in range(2):
                    A = CT["A"][ir,iex,ik,:,id_boot].sum(axis=0)+0.0001
                    B = CT["B"][ir,iex,ik,:,id_boot].sum(axis=0)+0.0001
                    C = CT["C"][ir,iex,ik,:,id_boot].sum(axis=0)+0.0001
                    D = CT["D"][ir,iex,ik,:,id_boot].sum(axis=0)+0.0001
                    sco = score_ct(A,B,C,D)
                    sc_dist[ib,ir,iex,ik,:] = sco
    return sc_dist


  
