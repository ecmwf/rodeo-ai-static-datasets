#
# Derive SEEPS coefficients and Climatology for eah station of the ECA&D 
# 
# ECA&D precipitation file as input. 
#   "Daily precipitation amount RR" downloaded https://www.ecad.eu/dailydata/predefinedseries.php  
#    
# See ECA&D data policy here: https://knmi-ecad-assets-prd.s3.amazonaws.com/documents/ECAD_datapolicy.pdf
#
# As output: 
# 1. percentiles (1,2,..., 98,99)
# 2. equitable categories (for SEEPS)
# 3. list of stations (and metadata)

import os
import numpy as np
import pandas as pd

#location identifier
ll = lambda icnt: int(lats[icnt]*10000)*1000+lons[icnt]

# Climatology years
year_clim_i = 1991
year_clim_f = 2020

# Raw ECAD directory
path_input="/perm/mozb/RODEO/SEEPS4ALL/ECAD/"

# output directory
path_output="/perm/mozb/RODEO/SEEPS4ALL/DATA"
# create local directory
clim_years = f"{year_clim_i}-{year_clim_f}"
path_local = f"{path_output}/local_climate_{clim_years}"
if not os.path.exists(path_local):
    os.mkdir(path_local)

# Read ECAD file
info = pd.read_csv(f"{path_input}/stations.txt", sep=",", header=25, names=("STAID","STANAME","CN","LAT","LON","HGHT") )

# basic metadata
lats =  [ int(a[0:3])+int(a[4:6])/60+int(a[7:9])/6000 for a in info["LAT"] ]
lons =  [ int(a[0:4])+int(a[5:7])/60+int(a[8:10])/6000 for a in info["LON"] ]
elevs =  info["HGHT"].values
stnid_list = info["STAID"]

# dates
time_start = pd.Timestamp("%s0101"%year_clim_i)
time_end   = pd.Timestamp("%s1231"%year_clim_f)
ndates = len(pd.date_range(time_start,time_end,freq="1D"))

# Quality control parameter (QC1)
p99_0 = 200

#initialistaion
cols = ["STNID","LAT","LON","ELEV","AVAIL","MVAL","Q95","Q99"]
data_st = pd.DataFrame(columns=cols)

latlons = []
for icnt, stnid in enumerate(stnid_list):

    print(f"{icnt+1}/{len(stnid_list)} ... station {stnid}")
    
    # read station file
    filein = "%s/RR_STAID0%05d.txt"%(path_input,stnid) 
    data = pd.read_csv(filein, sep=",", header=22, names=("STAID","SOUID","DATE","RR","Q_RR") )

    #get preicpitation values and quality control values
    val =  np.array([ float(v)/10. for v in data["RR"] ])
    qc  =  np.array([ int(v) for v in data["Q_RR"] ])
    # remove pooor quality measurements 
    val [ qc == 9 ] = np.nan

    #QC 1
    idno = ( val > 5*p99_0*np.cos(lats[icnt]*np.pi/180) )
    val [ idno ] = np.nan
    #QC 2
    uni = np.unique(val)
    for bigv in uni[uni>120]:
        if (val == bigv).sum() >1:
            val[ val == bigv ] = np.nan

    #QC 3
    #proportion of zeros
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
        
    #play with time and define month       
    tim =  np.array([ pd.Timestamp(str(v)) for v in data["DATE"] ] )
    mon = np.array([ t.month for t in tim ] )

    dok = (tim >= time_start) & (tim <= time_end) & (np.isfinite(val))
    if len(val[dok]) == 0:
        print(f" ... No valid data for station {stnid} for specified period. ")
        continue 
    avail = dok.sum()/ndates*100
    print("availability:",int(avail),"%")

    #climatology: percentiles
    month  = pd.DataFrame(mon[dok], columns=["month",])
    clim_0 = pd.DataFrame(val[dok], columns=["value",])
    clim_0 = pd.concat([clim_0,month],axis=1)
    clim_g0 = clim_0.groupby("month")

    n1 = ["count"]
    c1 = clim_g0.count()
    for q in range(1,100):
        n1.append("q%s"%q)
        c1 = pd.concat([c1,clim_g0.quantile(q/100)],axis=1)
    c1 = c1.set_axis(n1,axis=1)

    #climatology: equitable categories
    t1 = 0.25 # by definition
    clim_t1  = pd.DataFrame((val[dok]>t1)*1., columns=["p1",])
    clim_t1  = pd.concat([clim_0,clim_t1],axis=1)
    clim_g1  = clim_t1.groupby("month")
    p1 = 1 - clim_g1.mean()["p1"]
    c2 = pd.concat([clim_g1.count()["p1"],p1*0+t1,p1],axis=1,keys=["count_all","t1","p1"])

    month2 = pd.DataFrame(mon[dok][val[dok]>t1], columns=["month",])
    clim_t2  = pd.DataFrame(val[dok][val[dok]>t1], columns=["t2",])
    clim_t2  = pd.concat([clim_t2,month2],axis=1)
    clim_g2 = clim_t2.groupby("month")
    t2_m     = clim_g2.quantile(0.666)
    count_t1 = clim_g2.count().rename(columns={"t2":"count_t1"})
    c2 = pd.concat([c2, count_t1,t2_m ],axis=1)
        
    # A month with not enough data?  
    if len(t2_m.index) < 12 :
        print(f" ... No info for the 12 months of the year for station {stnid}. ")
        
    t2 = [ c2.loc[im].t2 for im in mon[dok]]

    clim_p2 = pd.DataFrame(val[dok]>t2, columns=["p2",])
    #clim_t2  = pd.concat([clim_t2,clim_p2],axis=1)
    clim_t2 = pd.concat([month,clim_p2],axis=1)
    clim_g2 = clim_t2.groupby("month")
    
    p2 = 1 - clim_g2.mean()["p2"]
    c2 = pd.concat([c2,p2],axis=1)

    # metadata type of info
    mval = np.nanmean(val[dok])
    q95  = np.nanquantile(val[dok],0.95)
    q99  = np.nanquantile(val[dok],0.99)
        
    new = {"STNID":int(stnid),"LAT":lats[icnt],"LON":lons[icnt],"ELEV":elevs[icnt],"AVAIL":avail,"MVAL":mval,"Q95":q95,"Q99":q99}
    data_st_new = pd.DataFrame(new,index=[int(stnid)])
    data_st = pd.concat([data_st,data_st_new],ignore_index=True)

    # ARCHIVE 
    c1.to_csv("%s/local_climate_%s/percentiles_st%s_%s.csv"%(path_output,clim_years,stnid,clim_years))
    c2.to_csv("%s/local_climate_%s/equitable_categories_st%s_%s.csv"%(path_output,clim_years,stnid,clim_years))
    data_st.to_csv("%s/local_climate_%s/stations_info_%s.csv"%(path_output,clim_years,clim_years))
  
# the end
