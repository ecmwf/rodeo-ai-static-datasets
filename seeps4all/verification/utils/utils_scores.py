"""Scoring routines for SEEPS4ALL.

Score functions to compute verification metrics and domain selection 
"""

import time
import copy
import numpy as np
import pandas as pd
import xarray as xr

def param_epsilon() -> float:
    """not zero, rather epsilon for threshold definition."""
    epsilon = 0.1
    return epsilon

def get_score(obs_seeps_data,
                  fct_data,
                  weights,
                  score):

    #get dimensions
    nexp = len(fct_data)
    nruns_ex,nsteps_ex = [],[]
    for iex in range(nexp):
        nruns_ex.append(len(fct_data[iex].run.values))
        nsteps_ex.append(len(fct_data[iex].step.values))
    nruns  = np.max(np.array(nruns_ex))
    nsteps  = np.max(np.array(nsteps_ex))

    #initialisation
    sc_all = np.zeros((2,nexp,nsteps,nruns))*np.nan

    for iex in range(nexp):
        runs = fct_data[iex].run.values
        steps = fct_data[iex].step.values

        print(f"experiment {iex}, {len(steps)} steps")
        
        #Loop over forecast lead times (steps) 
        print("step:",end=" ")
        for istep,step in enumerate(steps):
            print(istep,end=" ")

            vtime = [pd.Timestamp(vt) for vt in runs+step ]
            valid_t = ["%s-%02d-%02d"%(vt.year,vt.month,vt.day) for vt in vtime]
            fct_sel = fct_data[iex].sel(step=step)    
            data = obs_seeps_data.sel(time=valid_t)
            data = data.assign(forecast=fct_sel["forecast"])

            sc,sc_ref = compute_score_avg(data,weights,ens=fct_sel,score=score)
            sc_all[0,iex,istep,:] = sc
            sc_all[1,iex,istep,:] = sc_ref
            
        print("")
        
    return sc_all

def get_seeps_comps(obs_seeps_data,fct_data,weights):
    
    #get dimensions
    nexp = len(fct_data)
    nruns_ex,nsteps_ex = [],[]
    for iex in range(nexp):
        nruns_ex.append(len(fct_data[iex].run.values))
        nsteps_ex.append(len(fct_data[iex].step.values))
    nruns  = np.max(np.array(nruns_ex))
    nsteps  = np.max(np.array(nsteps_ex))

    #initialisation
    seeps_all = dict()
    for iex in range(nexp):
        seeps_all["%s"%(iex)] = []
    
    for iex in range(nexp):
        runs = fct_data[iex].run.values
        steps = fct_data[iex].step.values

        print(f"experiment {iex}, {len(steps)} steps")
        
        print("step:",end=" ")
        for istep,step in enumerate(steps):
            print(istep,end=" ")

            # validity date
            vtime = [pd.Timestamp(vt)-pd.Timedelta(24,unit="h") for vt in runs+step ]
            valid_t = ["%s-%02d-%02d"%(vt.year,vt.month,vt.day) for vt in vtime]

            # select forecast
            fct_sel = fct_data[iex].sel(step=step)

            # select observations
            data = obs_seeps_data.sel(time=valid_t)
            data = data.assign(forecast=fct_sel["forecast"])
        
            # seeps components
            sc = decompose_seeps(data,weights)
            seeps_all["%s"%(iex)].append(sc)
        print("")
        
    seeps_all["nexp"] = nexp
    seeps_all["nsteps"] = len(steps) 
    return seeps_all



def get_CT(obs_clim_data,
                  fct_data,
                  weights,thresholds):
    
    #get dimensions
    nthr = len(thresholds)
    nexp = len(fct_data)
    nruns_ex,nsteps_ex = [],[]
    for iex in range(nexp):
        nruns_ex.append(len(fct_data[iex].run.values))
        nsteps_ex.append(len(fct_data[iex].step.values))
    nruns  = np.max(np.array(nruns_ex))
    nsteps  = np.max(np.array(nsteps_ex))

    # initialise
    CT = dict()
    for n in ("A","B","C","D"):
        CT[n] = np.zeros((2,nexp,nthr,nsteps,nruns))
    
    for iex in range(nexp):
        runs = fct_data[iex].run.values
        steps = fct_data[iex].step.values

        print(f"experiment {iex}, {len(steps)} steps")
        
        #Loop over forecast lead times (steps) 
        print("step:",end=" ")
        for istep,step in enumerate(steps):
            print(istep,end=" ")

            # validity time
            vtime = [pd.Timestamp(vt) for vt in runs+step ]
            valid_t = ["%s-%02d-%02d"%(vt.year,vt.month,vt.day) for vt in vtime]

            # select forecast
            fct_sel = fct_data[iex].sel(step=step)

            # select observation and metadata
            data = obs_clim_data.sel(time=valid_t)
            data = data.assign(forecast=fct_sel["forecast"])

            for ik,thr in enumerate(thresholds):
                # contingency table
                a,b,c,d = compute_cts(data,weights,thr)
                CT["A"][0,iex,ik,istep,:]= a 
                CT["B"][0,iex,ik,istep,:]= b
                CT["C"][0,iex,ik,istep,:]= c 
                CT["D"][0,iex,ik,istep,:]= d 
        print("")
        
    return CT

def get_scores_thr(obs_data,fct_data,weights,thresholds):

    #get dimensions
    nthr = len(thresholds)
    
    nexp = len(fct_data)
    nruns_ex,nsteps_ex = [],[]
    for iex in range(nexp):
        nruns_ex.append(len(fct_data[iex].run.values))
        nsteps_ex.append(len(fct_data[iex].step.values))
    nruns  = np.max(np.array(nruns_ex))
    nsteps  = np.max(np.array(nsteps_ex))

    # initialise
    eds   = np.zeros((2,nexp,nthr,nsteps,nruns)) # elementary diagonal score
    bs    = np.zeros((2,nexp,nthr,nsteps,nruns)) # brier score
    
    for iex in range(nexp):
        runs = fct_data[iex].run.values
        steps = fct_data[iex].step.values

        print(f"experiment {iex}, {len(steps)} steps")
        
        print("step:",end=" ")
        for istep,step in enumerate(steps):
            print(istep,end=" ")

            # validity date
            vtime = [pd.Timestamp(vt)-pd.Timedelta(24,unit="h") for vt in runs+step ]
            valid_t = ["%s-%02d-%02d"%(vt.year,vt.month,vt.day) for vt in vtime]

            # select observations
            data = obs_data.sel(time=valid_t)

            # select ensemble
            ens = fct_data[iex].sel(step=step)

            # compute elementary diagonal score and Brier score for various threshold
            for ik,thr in enumerate(thresholds):

                # Brier score
                bscore,bscore_ref = compute_score_avg(data,weights,ens=ens,score="bs",threshold=thr)
                bs[0,iex,ik,istep,:]= bscore 
                bs[1,iex,ik,istep,:]= bscore_ref

                # elementary diagonal scores
                escore,escore_ref = compute_score_avg(data,weights,ens=ens,score="eds",threshold=thr)
                eds[0,iex,ik,istep,:]= escore 
                eds[1,iex,ik,istep,:]= escore_ref
        print("")

    return bs,eds


def compute_cts(data:xr.Dataset,
                weights:np.ndarray,
                thr: str,
                station_wise:bool=False
                )-> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """ Populate contingency tables. """

    ob = data.observation.values
    fc = data.forecast.values
    th = data[thr].values+param_epsilon()

    # mask
    mask = np.array(np.isfinite(th) & np.isfinite(fc)& np.isfinite(ob)) 

    # forecast yes/no 
    F1 = np.zeros(fc.shape)
    F1[fc >= th] = 1
    F0 = 1-F1

    # observed yes/no 
    G1 = np.zeros(fc.shape)
    G1[ob >= th] = 1
    G0 = 1-G1
    
    # 4 elements of the table
    A = F1 * G1 
    B = F1 * G0 
    C = F0 * G1 
    D = F0 * G0

    A[np.where(~mask)]=np.nan
    B[np.where(~mask)]=np.nan
    C[np.where(~mask)]=np.nan
    D[np.where(~mask)]=np.nan

    if station_wise:
        return A,B,C,D
        
    w = np.repeat(weights,fc.shape[0]).reshape(np.flip(fc.shape)).T
    w[np.where(~mask)]=np.nan
    mean_weights = np.nanmean(w[mask])
    
    A =  np.nansum(A*weights,axis=1)/mean_weights
    B =  np.nansum(B*weights,axis=1)/mean_weights
    C =  np.nansum(C*weights,axis=1)/mean_weights
    D =  np.nansum(D*weights,axis=1)/mean_weights
    
    return A,B,C,D

def fbi_ct(A:np.ndarray,B:np.ndarray,
           C:np.ndarray,D:np.ndarray)-> np.ndarray:
    """ FBI based on contingency table entries."""
    res = (A+B)/(A+C)
    return res

def ets_ct(A:np.ndarray,B:np.ndarray,
           C:np.ndarray,D:np.ndarray)-> np.ndarray:
    """ ETS based on contingency table entries."""
    H_rnd = (A+C)*(A+B)/(A+B+C+D)
    res = (A-H_rnd)/(A+B+C-H_rnd)
    return res
    
def pss_ct(A:np.ndarray,B:np.ndarray,
           C:np.ndarray,D:np.ndarray)-> np.ndarray:
    """ PSS based on contingency table entries."""
    POD =  A/(A+C)
    POFD = B/(B+D)
    res = POD-POFD 
    return res

def ora_ct(A:np.ndarray,B:np.ndarray,
           C:np.ndarray,D:np.ndarray)-> np.ndarray:
    """ ORA based on contingency table entries."""
    POD =  A/(A+C)
    POFD = B/(B+D)
    res = POD/(1-POD)*(1-POFD)/POFD 
    return res
    
def hss_ct(A:np.ndarray,B:np.ndarray,
           C:np.ndarray,D:np.ndarray)-> np.ndarray:
    """ HSS based on contingency table entries."""
    N = A+B+C+D
    exp_corr = ( (A+C)*(A+B)+(D+C)*(D+B) )/N
    res = (A+D-exp_corr)/(N-exp_corr) 
    return res

def sc_avg(sc: np.ndarray,
           weights: np.ndarray,
           nboot: int=1)-> np.ndarray:
    """ Score averaging with weights."""
    
    if np.isfinite(sc).sum()==0:
        return np.nan
        
    w = weights.reshape((1,len(weights)))
    sc_station  = np.mean(sc,axis=0)
    ido = np.isfinite(sc_station) 
    sc =  np.mean(sc[:,ido]*w[:,ido],axis=1)/np.mean(w[:,ido])
    
    return sc


def compute_score_avg(data:xr.Dataset,
                      weights:np.ndarray,
                      ens=[],
                      score: str="seeps",
                      threshold:str="perc99",)-> tuple[np.ndarray,np.ndarray]:
    """ Compute score and average it."""
    
    # scores
    if score == "seeps":
        sc,_ = compute_seeps(data)
        sc_ref = sc*np.nan
        
    elif score == "bs":
        sc, sc_ref = compute_bs(data,ens,threshold)

    elif score == "eds":
        sc, sc_ref = compute_elementary_diags(data,ens,threshold)

    elif score == "diags":
        sc, sc_ref = compute_diags(data,ens)

    # averaging over stations + weighting   
    sc     = sc_avg(sc,weights)
    sc_ref = sc_avg(sc_ref,weights)

    return sc,sc_ref
    
def compute_bs(data:xr.Dataset,
               ens:xr.Dataset,
               threshold:str)-> tuple[xr.Dataset,xr.Dataset]:
    """ Definition of the Brier score. """
    
    ob = data["observation"].values.flatten()
    fc = ens["forecast"].values
    th = data[threshold].values.flatten()+param_epsilon()

    o = (ob >= th )
    
    fc = fc.reshape((fc.shape[0],fc.shape[1]*fc.shape[2]))
    th = th.reshape((1,len(th)))
    f =  (fc >= th ).mean(axis=0)
    
    bs = (f - o )**2

    f_clim = 1 - int(threshold[4:])/100
    bs_ref = (f_clim - o )**2

    return bs.reshape(data.observation.shape),bs_ref.reshape(data.observation.shape)
 
def compute_elementary_diags(data,ens,threshold):
    """Definition of the elementary diagonal score 
    (DIAGS, Ben Bouallegue et al 2018, 
    https://doi.org/10.1002/qj.3293). """ 

    ob = data["observation"].values.flatten()
    fc = ens["forecast"].values    
    th = data[threshold].values.flatten()+param_epsilon()
    
    tau = int(threshold[4:])/100
    
    obs_ev = (ob > th)*1

    fc = fc.reshape((fc.shape[0],fc.shape[1]*fc.shape[2]))
    th = th.reshape((1,len(th)))
    p = (fc > th ).mean(axis=0).flatten()
    
    ds = obs_ev*(p <=(1.-tau))*tau + (1.-obs_ev)*(p>(1.-tau))*(1.-tau)

    p = 1 - int(threshold[4:])/100
    ds_ref = obs_ev*(p <=(1.-tau))*tau + (1.-obs_ev)*(p>(1.-tau))*(1.-tau)

    return ds.reshape(data.observation.shape),ds_ref.reshape(data.observation.shape)

def compute_diags(data:xr.Dataset,ens:xr.Dataset)-> tuple[xr.Dataset,xr.Dataset]:
#     """ Computes the diagonal score as 
#     the average diagonal elementary score   
#     over all unique climate quantile levels."""   

    # array with the ensemble members    
    fct = ens.forecast.values
    fct = fct.reshape((fct.shape[0],fct.shape[1]*fct.shape[2]))

    # observation value 
    obs = data.observation.values.flatten()
    obs = obs.reshape((1,len(obs)))

    # array of quantile levels corresponding to qclim 
    tau_clim = [ q/100 for q in range(1,100) ]
    nq = len(tau_clim)

    #  array of equidistant increasing quantiles   
    #  representing the climatology 
    qclim = []
    for iq in range(nq):
        qclim.append(data["perc%s"%(iq+1)].values.flatten())
    for iq in range(nq):
        qclim[iq] = qclim[iq].reshape((1,len(qclim[iq])))
        
    mask = []
    perc_ref = -np.inf
    for iq in range(nq):
        pos_mask = qclim[iq] > perc_ref
        perc_ref = qclim[iq]
        mask.append(pos_mask)
    perc_ref = np.inf
    for iq in range(nq)[::-1]:
        pos_mask = qclim[iq] < perc_ref
        perc_ref = qclim[iq]
        mask[iq] &= pos_mask
    nc = 0.
    dse,dse_ref = 0.,0.
    for iq in range(nq):
        tau = tau_clim[iq]
        obs_ev = obs > qclim[iq]
        pre_ev = fct > qclim[iq]
        
        p = pre_ev.mean(axis=0)
        dst = obs_ev*(p <=(1.-tau))*tau + (1.-obs_ev)*(p>(1.-tau))*(1.-tau)
        dse += dst*mask[iq]

        p_ref = 0 
        dst_ref = obs_ev*(p_ref <=(1.-tau))*tau + (1.-obs_ev)*(p_ref>(1.-tau))*(1.-tau)
        dse_ref += dst_ref*mask[iq]

        nc += mask[iq]

    idok= (nc>0)
    ds,ds_ref = obs*np.nan,obs*np.nan
    ds[idok] = 2.*dse[idok]/nc[idok]
    ds_ref[idok] = 2.*dse_ref[idok]/nc[idok]
    
    return ds.reshape(data.observation.shape),ds_ref.reshape(data.observation.shape)

def compute_seeps(data:xr.Dataset)-> tuple[xr.Dataset,xr.Dataset]:
    """" Definition of the stable and equitable error 
    in probability space (SEEPS, Rodwell et al 2010,
    https://doi.org/10.1002/qj.656). """

    ob = data.observation.values.flatten()
    fc = data.forecast.values.flatten()

    p1 = data.p1.values.flatten()
    t1 = data.t1.values.flatten()
    t2 = data.t2.values.flatten()

    seeps_mask = np.array( (1-p1 >= 0.1) & (1-p1 < 0.85) & np.isfinite(fc) & np.isfinite(ob))    
    
    mx = np.zeros((9,len(ob)))
    for k in range(9):
        mx[k,:] = data["mx%s"%(k+1)].values.flatten()

    ob_ind = (ob > t1).astype(int) + (ob >= t2).astype(int)
    fc_ind = (fc > t1).astype(int) + (fc >= t2).astype(int)
    indices = fc_ind * 3 + ob_ind
    seeps = np.array([ mx[idx,jj] for jj,idx in enumerate(indices)])
 
    seeps[~seeps_mask] = np.nan

    seeps = seeps.reshape(data.observation.shape)
    indices = indices.reshape(data.observation.shape)

    return seeps,indices

def decompose_seeps(data:xr.Dataset,
                    weights:np.ndarray)-> dict:  

    """ SEEPS decomposition. """

    w_fac = weights/ np.nanmean(weights)
    w_fac = np.repeat(w_fac,len(data.time)).reshape((len(weights),len(data.time))).T
    
    # compute seeps
    seeps_all,indices = compute_seeps(data)
    
    seeps_comp = dict()
    for icat in range(9):
        idd = (indices == icat)

        ncount = np.nansum(idd)
        seeps_c = 0
        if ncount > 0:
            seeps_c = np.nanmean(seeps_all[idd],axis=0)                     
        seeps_comp[f"seeps{icat}"] = seeps_c  
        seeps_comp[f"cnt{icat}"] = ncount 

    #BASE RATE info
    p1 = data["p1"].values.flatten()*w_fac.flatten()
    p2 = data["p2"].values.flatten()*w_fac.flatten()
    
    br1 = np.nanmean(1 - p1)
    seeps_comp["br1"] = br1  
    br2 = np.nanmean(p1 - p2)
    seeps_comp["br2"] = br2  
    br3 = np.nanmean(p2)
    seeps_comp["br3"] = br3  

    ob = data.observation.values.flatten()
    fc = data.forecast.values.flatten()
    t1 = data.t1.values.flatten()
    t2 = data.t2.values.flatten()

    ob_ind = (ob > t1).astype(int) + (ob >= t2).astype(int)
    fc_ind = (fc > t1).astype(int) + (fc >= t2).astype(int)
    for icat in range(3):
        fqo = np.nanmean((ob_ind == icat)  )
        seeps_comp[f"fqo{icat+1}"] = fqo
        fqf = np.nanmean((fc_ind == icat) )
        seeps_comp[f"fqf{icat+1}"] = fqf

    return seeps_comp

