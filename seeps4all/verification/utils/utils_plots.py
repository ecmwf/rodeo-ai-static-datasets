"""Plotting routines for SEEPS4ALL.

Plot as a function of lead time or threshold, plot score or skill score.
"""

import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from utils.utils_stats import simple_bootstrap
from utils.utils_stats import ct_bootstrap
from utils.utils_stats import sc_bootstrap

def plot_scores(score_data:xr.Dataset,
                score_name:str,
                plot_info:dict,
                along:str="steps",
                bootstrap:bool=True,
                x_list:list=()
                )-> None:

    """ Plot scores ... but bootstrap first! """

    # boostrapped scores
    sce = score_name.split(" ")
    if ("Brier" in sce) | ("Diagonal" in sce):
        sc_dist = sc_bootstrap (score_data,bootstrap)
    else:
        sc_dist = ct_bootstrap (score_data,score_name,bootstrap)

    if along in ("step","steps"):
        plot_scores_along_steps(
                    sc_dist,score_name,plot_info,
                    ithr_list  = x_list
        )
    elif along in ("threshold","thresholds"):
        plot_scores_along_thresholds(
                    sc_dist,score_name,plot_info,
                    istep_list = x_list
        )
    else:
        raise NameError(f"please define axis {along}")

def plot_scores_along_steps(sc_dist:np.ndarray,
                            score_name:str,
                            plot_info=dict,
                            ithr_list:list=(),
                           )-> None:
    """ Plot as a function fo the lead time (for  a given threshold)"""
    
    colors = plot_info["colors"]
    labels = plot_info["labels"]
    prefig = plot_info["prefig"]
    thresholds  = plot_info["thresholds"]

    nexp   = sc_dist.shape[2]
    nsteps = sc_dist.shape[4]

    # all thresholds (no particular selection)
    if len(ithr_list) == 0 :
        ithr_list = range(len(thresholds))
    
    ne = len(ithr_list)    
    if ne == 1:
        fig, axs = plt.subplots(1,ne,figsize=(4.5,3.5))
    else:
        fig, axs = plt.subplots(1,ne,figsize=(12,3.5))
                                
    x = np.array(range(1,nsteps+1))

    if len(thresholds) == 1:
        axs = [axs] 
        
    for ix,ik in enumerate(ithr_list):
        thr = thresholds[ik]
        for iex in range(nexp):

            if "skill" in score_name.split(" "):
                if nexp == 1:
                    sc_m = 1-np.mean(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],axis=0)
                    sc_l = 1-np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],0.025,axis=0)
                    sc_h = 1-np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],0.975,axis=0)
                    axs[ix].plot(x.astype(float),x.astype(float)*0,"--",color="grey")
                else:
                    sc_m = 1-np.mean(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],axis=0)
                    sc_l = 1-np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],0.025,axis=0)
                    sc_h = 1-np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],0.975,axis=0)
            elif "gain" in score_name.split(" "):
                if nexp == 1:
                    sc_m = np.mean(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],axis=0)-1
                    sc_m = np.mean(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],axis=0)-1
                    sc_l = np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],0.025,axis=0)-1
                    sc_h = np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,1,iex,ik,:],0.975,axis=0)-1
                    axs[ix].plot(x.astype(float),x.astype(float)*0,"--",color="grey")
                else:
                    sc_m = np.mean(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],axis=0)-1
                    sc_m = np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],0.500,axis=0)-1
                    sc_l = np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],0.025,axis=0)-1
                    sc_h = np.quantile(sc_dist[:,0,iex,ik,:]/sc_dist[:,0,0,ik,:],0.975,axis=0)-1

            elif "difference" in score_name.split(" "):
                sc_m = np.mean(sc_dist[:,0,iex,ik,:]-sc_dist[:,0,0,ik,:],axis=0)
                sc_l = np.quantile(sc_dist[:,0,iex,ik,:]-sc_dist[:,0,0,ik,:],0.025,axis=0)
                sc_h = np.quantile(sc_dist[:,0,iex,ik,:]-sc_dist[:,0,0,ik,:],0.975,axis=0)
            else:   
                sc_m = np.mean(sc_dist[:,0,iex,ik,:],axis=0)
                sc_l = np.quantile(sc_dist[:,0,iex,ik,:],0.025,axis=0)
                sc_h = np.quantile(sc_dist[:,0,iex,ik,:],0.975,axis=0)
                
                
            axs[ix].set_ylabel(score_name)
            if score_name == "FBI":
                axs[ix].plot(x,x.astype(float)*0+1,"--",color="grey")
           
            #axs[ik].plot(x, sco[0,iex,ik,:,].mean(axis=1),"-s",color=colors[iex],label=labels[iex])
            axs[ix].plot(x,sc_m,"-o",color=colors[iex],label=labels[iex])
            
            axs[ix].fill_between(x.astype(float),sc_l,sc_h,color=colors[iex],alpha=0.35)
            axs[ix].set_xlabel("lead time [d]")
        axs[ix].set_title(f"{thr[4:]}%-percentile climatology")
        axs[ix].grid(color= "grey", linestyle='--', linewidth=0.22 )
    axs[0].legend()
    plt.tight_layout()

    if prefig != None:
        if "skill" in score_name.split(" "):
            score_name_output = f"{score_name[0:3].lower()}_skill"
        elif "gain" in score_name.split(" "):
            score_name_output = f"{score_name[0:3].lower()}_gain"
        elif "difference" in score_name.split(" "):
            score_name_output = f"{score_name[0:3].lower()}_difference"
        else:    
            score_name_output = score_name[0:3].lower()
        
        file_output = f"{prefig}_{score_name_output}_fct_leadtime.png"
        plt.savefig(file_output)

def plot_scores_along_thresholds(sc_dist:np.ndarray,
                            score_name:str,
                            plot_info=dict,
                            istep_list:list=(),
                           )-> None:
    """ Plot as a function of the threshold (for a given lead time or step)."""
    
    colors = plot_info["colors"]
    labels = plot_info["labels"]
    prefig = plot_info["prefig"]
    thresholds = plot_info["thresholds"]

    nexp   = sc_dist.shape[2]
    nsteps = sc_dist.shape[4]
    
    # all steps (no particular selection)
    if len(istep_list) == 0 :
        istep_list = range(nsteps) 

    ne = len(istep_list)    
    if ne == 1:
        fig, axs = plt.subplots(1,ne,figsize=(4.5,3.5))
    else :
        fig, axs = plt.subplots(1,ne,figsize=(12,3.5))
    steps = np.array(range(1,nsteps+1))
    x = steps

    if len(istep_list) == 1:
        axs = [axs] 

    thr = np.array([ int(k[4:]) for k in thresholds])
    x = np.array(range(len(thr)))
    
    for ix,it in enumerate(istep_list):
        for iex in range(nexp):
            if "skill" in score_name.split(" "):
                if nexp == 1:
                    sc_m = 1-np.mean(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],axis=0)
                    sc_l = 1-np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],0.025,axis=0)
                    sc_h = 1-np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],0.975,axis=0)
                    axs[ix].plot(x.astype(float),x.astype(float)*0,"--",color="grey")
                else:
                    sc_m = 1-np.mean(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],axis=0)
                    sc_l = 1-np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],0.025,axis=0)
                    sc_h = 1-np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],0.975,axis=0)
            elif "gain" in score_name.split(" "):
                if nexp == 1:
                    #sc_m = np.mean(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],axis=0)-1
                    sc_m = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],0.500,axis=0)-1
                    sc_l = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],0.025,axis=0)-1
                    sc_h = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,1,iex,:,it],0.975,axis=0)-1
                    axs[ix].plot(x.astype(float),x.astype(float)*0,"--",color="grey")
                else:
                    #sc_m = np.mean(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],axis=0)
                    sc_m = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],0.500,axis=0)-1
                    sc_l = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],0.025,axis=0)-1
                    sc_h = np.quantile(sc_dist[:,0,iex,:,it]/sc_dist[:,0,0,:,it],0.975,axis=0)-1
            elif "difference" in score_name.split(" "):
                sc_m = np.mean(sc_dist[:,0,iex,:,it]-sc_dist[:,0,0,:,it],axis=0)
                sc_l = np.quantile(sc_dist[:,0,iex,:,it]-sc_dist[:,0,0,:,it],0.025,axis=0)
                sc_h = np.quantile(sc_dist[:,0,iex,:,it]-sc_dist[:,0,0,:,it],0.975,axis=0)
            else:   
                sc_m = np.mean(sc_dist[:,0,iex,:,it],axis=0)
                sc_l = np.quantile(sc_dist[:,0,iex,:,it],0.025,axis=0)
                sc_h = np.quantile(sc_dist[:,0,iex,:,it],0.975,axis=0)
                
            axs[ix].set_ylabel(score_name)
            if score_name == "FBI":
                axs[ix].plot(x,x*0+1,"--",color="grey")
           
            axs[ix].plot(x,sc_m,"-o",color=colors[iex],label=labels[iex])
            
            axs[ix].fill_between(x,sc_l,sc_h,color=colors[iex],alpha=0.35)
            axs[ix].set_xticks(ticks=x,labels=thr)
            axs[ix].set_xlabel("local climate percentile [%]")
        #
        axs[ix].set_title(f" day {steps[it]}")
        axs[ix].grid(color= "grey", linestyle='--', linewidth=0.22 )
    axs[0].legend()
    plt.tight_layout()

    if prefig != None:
        scn = score_name.split(" ")[0].lower()
        if "skill" in score_name.split(" "):
            score_name_output = f"{scn}_skill"
        elif "gain" in score_name.split(" "):
            score_name_output = f"{scn}_gain"
        elif "difference" in score_name.split(" "):
            score_name_output = f"{scn}_difference"
        else:    
            score_name_output = score_name[0:3].lower()
        file_output = f"{prefig}_{score_name_output}_fct_threshold.png"
        plt.savefig(file_output)

def plot_simple_scores(sco:np.ndarray,
                       score_name:str,
                       plot_info=dict,
                       bootstrap:bool=True
                      )-> None:
    """ Plot simple scores (no thresholds)."""
    
    prefig = plot_info["prefig"]
    colors = plot_info["colors"]
    labels = plot_info["labels"]

    nexp   = sco.shape[1]
    nsteps  = sco.shape[2]
    
    # bootstrap
    sc_dist = simple_bootstrap (sco,bootstrap)
    
    fig, axs = plt.subplots(figsize=(4.5,3.5))
    #x = steps.astype('timedelta64[h]')/24
    x = np.array(range(1,nsteps+1))

    for iex in range(nexp):
        if "skill" in score_name.split(" "):
            if nexp == 1:
                sc_m = 1-np.mean(sc_dist[:,0,iex,:]/sc_dist[:,1,iex,:],axis=0)
                sc_l = 1-np.quantile(sc_dist[:,0,iex,:]/sc_dist[:,1,iex,:],0.025,axis=0)
                sc_h = 1-np.quantile(sc_dist[:,0,iex,:]/sc_dist[:,1,iex,:],0.975,axis=0)
                axs.plot(x.astype(float),x.astype(float)*0,"--",color="grey")
            else:
                sc_m = 1-np.mean(sc_dist[:,0,iex,:]/sc_dist[:,0,0,:],axis=0)
                sc_l = 1-np.quantile(sc_dist[:,0,iex,:]/sc_dist[:,0,0,:],0.025,axis=0)
                sc_h = 1-np.quantile(sc_dist[:,0,iex,:]/sc_dist[:,0,0,:],0.975,axis=0)
        else:
            sc_m = np.mean(sc_dist[:,0,iex,:],axis=0)
            sc_l = np.quantile(sc_dist[:,0,iex,:],0.025,axis=0)
            sc_h = np.quantile(sc_dist[:,0,iex,:],0.975,axis=0)
    
        axs.plot(x,sc_m,"-o",color=colors[iex],label=labels[iex])
    
        axs.fill_between(x.astype(float),sc_l,sc_h,color=colors[iex],alpha=0.35)
    axs.set_xlabel("lead time [d]")
    axs.set_ylabel(score_name)
    plt.grid(color= "grey", linestyle='--', linewidth=0.22 )
    plt.legend()
    plt.tight_layout()
    if prefig != None:
        scn = score_name.split(" ")[0].lower()
        if "skill" in score_name.split(" "):
            file_output=f"{prefig}_{scn}_skillscore.png"  
        else:
            file_output=f"{prefig}_{scn}.png" 
        plt.savefig(file_output)
        
def data_plot_seeps_comp(sc_all:np.ndarray,
                         steps:np.ndarray,
                         axs:np.ndarray,
                         iex:int=0,
                         labf:str="forecsat",
                         colf:str="tab:red")->tuple:

    """ Plot SEEPS components (3x3 categories)."""
    
    nsteps = len(steps)
    
    # initialisation 
    allc = np.zeros((nsteps,9))
    numc = np.zeros((nsteps,9))
    brac = np.zeros((nsteps,9))
    for istep,step in enumerate(steps):
        sc =sc_all[istep]
        
        #compute averages and base rates per lead time
        for icat in range(9):
            if sc[f"cnt{icat}"].sum() > 0 :
                allc[istep,icat] = (sc[f"seeps{icat}"]*sc[f"cnt{icat}"]).sum()/sc[f"cnt{icat}"].sum()
                numc[istep,icat] = sc[f"cnt{icat}"].sum()

        for icat in range(3):
            brac[istep,icat]   = sc[f"br{icat+1}"].mean()
            brac[istep,icat+3] = sc[f"fqo{icat+1}"].mean()
            brac[istep,icat+6] = sc[f"fqf{icat+1}"].mean()

        all_cnt = 0
        for icat in range(9):
             all_cnt += sc[f"cnt{icat}"].sum()
        for icat in range(3):
            brac[istep,6] = (sc[f"cnt{0}"]+ sc[f"cnt{1}"]+ sc[f"cnt{2}"]).sum() /all_cnt
            brac[istep,7] = (sc[f"cnt{3}"]+ sc[f"cnt{4}"]+ sc[f"cnt{5}"]).sum() /all_cnt
            brac[istep,8] = (sc[f"cnt{6}"]+ sc[f"cnt{7}"]+ sc[f"cnt{8}"]).sum() /all_cnt
        
    # initialisation
    perc = []

    # loop over seeps categories (3x3 contingency table)
    for icat in range(9):
        x = steps
        y = allc[:,icat]*numc[:,icat]/numc.mean(axis=1)/9
        
        if np.max(y) >0:
            axs[int(icat/3),icat%3].plot(x, y,color=colf,linewidth=2)

        # % of cases for each category
        perc.append( numc[:,icat].sum()/numc.sum())

        if icat >5:
            axs[int(icat/3),icat%3].set_xlabel('lead time [d]',fontsize=12)
        else:
            axs[int(icat/3),icat%3].set_xticklabels(())
        axs[int(icat/3),icat%3].yaxis.set_tick_params(labelsize=8)
        axs[int(icat/3),icat%3].xaxis.set_tick_params(labelsize=8)
        axs[int(icat/3),icat%3].grid(color= "grey", linestyle='--', linewidth=0.22 )
        axs[int(icat/3),icat%3].set_ylabel('score',fontsize=10)

        if icat not in (2,6):
            yc = allc[:,icat]*numc[:,icat]/numc.mean(axis=1)/9
            yc = allc[:,icat]*numc[:,icat]
 
    # descending diagonal: base rates
    for icat in range(9):
        x = steps
        if icat in (0,4,8):
            y = brac[:,int(icat/3)]
            axs[int(icat/3),icat%3].plot(x, y,"-",color="grey",linewidth=1)

            y = brac[:,int(icat/3)+3]
            axs[int(icat/3),icat%3].plot(x, y,"--",color="k",linewidth=1)

            y = brac[:,int(icat/3)+6]
            axs[int(icat/3),icat%3].plot(x, y,"--",color=colf,linewidth=1)

            axs[int(icat/3),icat%3].set_ylim([0,1])
            axs[int(icat/3),icat%3].set_ylabel('base rate',fontsize=10)

        if (icat == 4):
            y = brac[:,int(icat/3)]
            if iex == 0:
                axs[int(icat/3),icat%3].plot(x, y,"-",label="clim",color="grey",linewidth=1)
            
            y = brac[:,int(icat/3)+3]
            if iex == 0:
                axs[int(icat/3),icat%3].plot(x, y,"--",label="obs",color="k",linewidth=1)
            axs[int(icat/3),icat%3].legend(fontsize=8)

        if icat == 8:
            y = brac[:,int(icat/3)+6]
            axs[int(icat/3),icat%3].plot(x, y+10,"-",label=labf,color=colf,linewidth=1)

    return perc


def plot_seeps_comp(sc_all:np.ndarray,
                    plot_info:dict
                   )-> None:
    
    prefig = plot_info["prefig"]
    colors = plot_info["colors"]
    labels = plot_info["labels"]
    
    nexp   = sc_all["nexp"]
    nsteps = sc_all["nsteps"]
    steps  = np.array(range(1,nsteps+1))
    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))

    perc_all = []
    for iex in range(nexp):
        perc = data_plot_seeps_comp(sc_all["%s"%(iex)],steps,axs,iex=iex,colf=colors[iex],labf=labels[iex])
        perc_all.append(perc)
    
    for iex in range(nexp):   
        # % of cases for each category
        for icat in range(9):
            xmin, xmax, ymin, ymax =  axs[int(icat/3),icat%3].axis()
            axs[int(icat/3),icat%3].set_ylim(ymin,ymax+(ymax-ymin)*0.2)
            xmin, xmax, ymin, ymax =  axs[int(icat/3),icat%3].axis()
            ytext = ymax - (ymax-ymin)*0.13
            xtext = xmax - (xmax-xmin)*0.94
            ttext = f"{np.round(perc_all[iex][icat]*100,1)}%"
            axs[int(icat/3),icat%3].text(xtext,ytext,ttext,color =colors[iex],fontsize=10,fontstyle="italic")

    plt.legend(fontsize=8)

    # add legends, labels, and text  
    pint = ["Dry","Light","Heavy"]
    for ip in range(3):
        xmin, xmax, ymin, ymax =  axs[0,ip].axis()
        xtext = xmin + (xmax-xmin)*0.42
        ytext = ymax + (ymax-ymin)*0.1
        axs[0,ip].text(xtext,ytext,pint[ip],fontsize=12)

        xmin, xmax, ymin, ymax =  axs[ip,0].axis()
        xtext = xmin - (xmax-xmin)*0.4
        ytext = ymin + (ymax-ymin)*0.36
        axs[ip,0].text(xtext,ytext,pint[ip],fontsize=12,rotation=90)

    xmin, xmax, ymin, ymax =  axs[0,1].axis()
    xtext = xmin + (xmax-xmin)*0.28 
    ytext = ymax + (ymax-ymin)*0.52
    axs[0,1].text(xtext,ytext,"Observed",fontsize=14)

    xmin, xmax, ymin, ymax =  axs[1,0].axis()
    xtext = xmin - (xmax-xmin)*0.60
    ytext = ymin + (ymax-ymin)*0.18
    axs[1,0].text(xtext,ytext,"Forecast",fontsize=14,rotation=90)

    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    fig.tight_layout()

    if prefig != None:
        file_ouput=f"{prefig}_seeps_components.png"
        plt.savefig(file_ouput)

