##!/usr/bin/env python
"""common_functions.py

Contains common functions for checking model simulation skill based upon the metrics detailed in Collier et al. [2018] doi:10.1029/2018MS001354

Author: Annette L Hirsch @ CLEX, UNSW. Sydney (Australia)
email: a.hirsch@unsw.edu.au
Created: Fri May 17 14:35:11 AEST 2019

"""

# Load packages

#from __future__ import division
import numpy as np
import netCDF4 as nc
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import path
import xarray
#from mpl_toolkits.basemap import Basemap
import math
import itertools
from scipy import ndimage


# Calculates the normalised bias error score
def calc_bias(mdata,odata,lat2d,flag=True):
    """This function calculates the bias score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        flag == logical to return the area-weighted average """

    nt = mdata.shape[0]
    
    # calculate the time-averaged mean 
    mmean = np.nanmean(mdata,axis=0)
    omean = np.nanmean(odata,axis=0)
        
    # calculate the bias
    bias = mmean - omean
    
    # calculate the centralised RMS
    crms = ((1/nt)*np.nansum((odata-omean)**2,axis=0))**(1/2)
    
    # calculate the relative bias error
    ebias = abs(bias)/crms
    
    # calculate the bias score
    sbias = np.exp(-ebias)

    if flag == True:
        # calculate the area-weighted average
        latr = np.deg2rad(lat2d)
        weights = np.cos(latr)
        sbias_wgt = np.ma.average(np.ma.MaskedArray(sbias, mask=np.isnan(sbias)),weights=weights)

        return sbias_wgt

    else:
        
        return sbias

# Calculates the normalised RMSE score
def calc_rmse(mdata,odata,lat2d,flag=True):
    """This function calculates the rmse score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        flag == logical to return the area-weighted average """

    nt = mdata.shape[0]
    
    # calculate the time-averaged mean 
    mmean = np.nanmean(mdata,axis=0)
    omean = np.nanmean(odata,axis=0)
        
    # calculate the rmse
#    crmse = ((1/nt)*np.nansum(((mdata-mmean)-(odata-omean))**2,axis=0))**(1/2)
    crmse = ((1/nt)*np.nansum(((np.ma.MaskedArray(mdata, mask=np.isnan(mdata))-mmean)-(np.ma.MaskedArray(odata, mask=np.isnan(odata))-omean))**2,axis=0))**(1/2)
    
    # calculate the centralised RMS
    crms = ((1/nt)*np.nansum((odata-omean)**2,axis=0))**(1/2)
    
    # calculate the relative RMSE error
    ermse = abs(crmse)/crms
    
    # calculate the RMSE score
    srmse = np.exp(-ermse)
    
    if flag == True:
        # calculate the area-weighted average
        latr = np.deg2rad(lat2d)
        weights = np.cos(latr)
        srmse_wgt = np.ma.average(np.ma.MaskedArray(srmse, mask=np.isnan(srmse)),weights=weights)

        return srmse_wgt

    else:
        
        return srmse

# Calculates the phase shift in timing of maxima
def calc_phase(mdata,odata,lat2d,flag=True):
    """This function calculates the phase shift in the timing of maxima using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        flag == logical to return the area-weighted average """

    nt = mdata.shape[0]
    
    # calculate the phase shift
#    theta = np.argmax(mdata,axis=0) - np.argmax(odata,axis=0)
    theta = np.argmax(np.ma.MaskedArray(mdata, mask=np.isnan(mdata)),axis=0) - np.argmax(np.ma.MaskedArray(odata, mask=np.isnan(odata)),axis=0)
    
    # calculate the phase shift score
    sphase = (0.5) * (1 + np.cos((2*math.pi*theta)/nt))
    
    if flag == True:
        # calculate the area-weighted average
        latr = np.deg2rad(lat2d)
        weights = np.cos(latr)
        sphase_wgt = np.ma.average(np.ma.MaskedArray(sphase, mask=np.isnan(sphase)),weights=weights)

        return sphase_wgt

    else:
        
        return sphase

# Calculates the spatial distribution score
def calc_spatialdsn(mdata,odata,lat2d):
    """This function calculates the spatial distribution score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average"""

    nt = mdata.shape[0]
    nxy = mdata.shape[1]*mdata.shape[2]
    
    # calculate the time-averaged mean 
    mmean = np.nanmean(mdata,axis=0)
    omean = np.nanmean(odata,axis=0)
    
    # calculate the normalised standard deviation
    mstd = np.nanstd(mmean)
    ostd = np.nanstd(omean)
    nstd = mstd / ostd
    
    # calculate the spatial correlation coefficient
    r = ((1/nxy)*np.nansum((mmean-np.nanmean(mmean))*(omean-np.nanmean(omean))))/(mstd*ostd)
    
    # calculate the spatial distribution score
    sdist = (2*(1+r))/((nstd + (1/nstd))**2)
        
    return sdist

# The following function was found on 20.05.2019    
#https://gis.stackexchange.com/questions/71630/subsetting-a-curvilinear-netcdf-file-roms-model-output-using-a-lon-lat-boundin
def bbox2ij(lon,lat,bbox=[-160., -155., 18., 23.]):
    """Return indices for i,j that will completely cover the specified bounding box.     
    i0,i1,j0,j1 = bbox2ij(lon,lat,bbox)
    lon,lat = 2D arrays that are the target of the subset
    bbox = list containing the bounding box: [lon_min, lon_max, lat_min, lat_max]

    Example
    -------  
    >>> i0,i1,j0,j1 = bbox2ij(lon_rho,[-71, -63., 39., 46])
    >>> h_subset = nc.variables['h'][j0:j1,i0:i1]       
    """
    
    bbox=np.array(bbox)
    mypath=np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(),lat.flatten())).T   
    n,m = np.shape(lon)
    inside = p.contains_points(points).reshape((n,m))
    ii,jj = np.meshgrid(range(m),range(n))
    return min(ii[inside]),max(ii[inside]),min(jj[inside]),max(jj[inside])


# Calculates the normalised bias error score for temperature extremes
def calc_bias_txtn(mdata,odata,lat2d,tx,flag=True):
    """This function calculates the bias score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        tx == calculate TX or TN extremes
        flag == logical to return the area-weighted average """

    nt = mdata.shape[0]
    nx = mdata.shape[1]
    ny = mdata.shape[2]
    
    mdata = np.ma.masked_array(mdata, mdata>=1.e20).filled(np.nan)
    odata = np.ma.masked_array(odata, odata>=1.e20).filled(np.nan)
    
    # calculate the extremes 
    m05 = np.nanpercentile(mdata,5,axis=0)
    m95 = np.nanpercentile(mdata,95,axis=0)
    o05 = np.nanpercentile(odata,5,axis=0)
    o95 = np.nanpercentile(odata,95,axis=0)
    if tx in ["TX"]:
        mex = np.nanmax(mdata,axis=0)
        oex = np.nanmax(odata,axis=0)
    if tx in ["TN"]:
        mex = np.nanmin(mdata,axis=0)
        oex = np.nanmin(odata,axis=0)
        
    # calculate the bias
    bias05 = m05 - o05
    bias95 = m95 - o95
    biasex = mex - oex
    
    # calculate the centralised RMS
    crms = ((1/nt)*np.nansum((odata-np.nanmean(odata,axis=0))**2,axis=0))**(1/2)
    
    # calculate the relative bias error
    ebias05 = abs(bias05)/crms
    ebias95 = abs(bias95)/crms
    ebiasex = abs(biasex)/crms
    
    # calculate the bias score
    sbias05 = np.exp(-ebias05)
    sbias95 = np.exp(-ebias95)
    sbiasex = np.exp(-ebiasex)

    if flag == True:
        # calculate the area-weighted average
        latr = np.deg2rad(lat2d)
        weights = np.cos(latr)
        sbias05_wgt = np.ma.average(np.ma.MaskedArray(sbias05, mask=np.isnan(sbias05)),weights=weights)
        sbias95_wgt = np.ma.average(np.ma.MaskedArray(sbias95, mask=np.isnan(sbias95)),weights=weights)
        sbiasex_wgt = np.ma.average(np.ma.MaskedArray(sbiasex, mask=np.isnan(sbiasex)),weights=weights)

        return [sbias05_wgt,sbias95_wgt,sbiasex_wgt]

    else:
        
        sbias = np.empty((3,nx,ny),dtype=np.float64)
        sbias[0,:,:] = sbias05
        sbias[1,:,:] = sbias95
        sbias[2,:,:] = sbiasex
        return sbias

# Calculates the spatial distribution score
def calc_spatialdsn_txtn(mdata,odata,lat2d,tx):
    """This function calculates the spatial distribution score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        tx == calculate TX or TN extremes"""

    nt = mdata.shape[0]
    nxy = mdata.shape[1]*mdata.shape[2]

    mdata = np.ma.masked_array(mdata, mdata>=1.e20).filled(np.nan)
    odata = np.ma.masked_array(odata, odata>=1.e20).filled(np.nan)
    
    # calculate the extremes 
    m05 = np.nanpercentile(mdata,5,axis=0)
    m95 = np.nanpercentile(mdata,95,axis=0)
    o05 = np.nanpercentile(odata,5,axis=0)
    o95 = np.nanpercentile(odata,95,axis=0)
    if tx in ["TX"]:
        mex = np.nanmax(mdata,axis=0)
        oex = np.nanmax(odata,axis=0)
    if tx in ["TN"]:
        mex = np.nanmin(mdata,axis=0)
        oex = np.nanmin(odata,axis=0)
            
    # calculate the normalised standard deviation
    nstd05 = np.nanstd(m05) / np.nanstd(o05)
    nstd95 = np.nanstd(m95) / np.nanstd(o95)
    nstdex = np.nanstd(mex) / np.nanstd(oex)
    
    # calculate the spatial correlation coefficient
    r05 = ((1/nxy)*np.nansum((m05-np.nanmean(m05))*(o05-np.nanmean(o05))))/(np.nanstd(m05)*np.nanstd(o05))
    r95 = ((1/nxy)*np.nansum((m95-np.nanmean(m95))*(o95-np.nanmean(o95))))/(np.nanstd(m95)*np.nanstd(o95))
    rex = ((1/nxy)*np.nansum((mex-np.nanmean(mex))*(oex-np.nanmean(oex))))/(np.nanstd(mex)*np.nanstd(oex))
    
    # calculate the spatial distribution score
    sdist05 = (2*(1+r05))/((nstd05 + (1/nstd05))**2)
    sdist95 = (2*(1+r95))/((nstd95 + (1/nstd95))**2)
    sdistex = (2*(1+rex))/((nstdex + (1/nstdex))**2)
        
    return [sdist05,sdist95,sdistex]

# The following found from: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def calc_mov_avg(data,N):
    """Calculates the N-day moving average"""
    cumsum, moving_aves = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
            
    return moving_aves

# Calculates the normalised bias error score for precipitation extremes
def calc_bias_pr(mdata,odata,lat2d,flag=True):
    """This function calculates the bias score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        flag == logical to return the area-weighted average """

    nt = mdata.shape[0]
    nx = mdata.shape[1]
    ny = mdata.shape[2]
    
    mdata = np.ma.masked_array(mdata, mdata>=1.e20).filled(np.nan)
    odata = np.ma.masked_array(odata, odata>=1.e20).filled(np.nan)
    
    # RX1DAY
    mrx1day = np.nanmax(mdata,axis=0)
    orx1day = np.nanmax(odata,axis=0)
    
    # RX5DAY
    mrx5day = np.nanmax(ndimage.uniform_filter(mdata, size=(5,0,0)),axis=0)
    orx5day = np.nanmax(ndimage.uniform_filter(odata, size=(5,0,0)),axis=0)

    # CDD
    mcdd = np.empty((nx,ny),dtype=np.float64)
    ocdd = np.empty((nx,ny),dtype=np.float64)
    mrain = np.where(mdata < 1., 1, 0) # set all days with rain < 1mm to 1 and 0 otherwise
    orain = np.where(odata < 1., 1, 0)
    
    for ii in range(nx):
        for jj in range(ny):
                    
            #https://stackoverflow.com/questions/22214086/python-a-program-to-find-the-length-of-the-longest-run-in-a-given-list
            mcdd[ii,jj] = max(sum(1 for _ in l) for n, l in itertools.groupby(mrain[:,ii,jj]))
            ocdd[ii,jj] = max(sum(1 for _ in l) for n, l in itertools.groupby(orain[:,ii,jj]))
          
    # R10mm - number of days with >10mm
    mr10mm = np.nansum(np.where(mdata >= 10., 1, 0),axis=0)
    or10mm = np.nansum(np.where(odata >= 10., 1, 0),axis=0)
    
    # calculate the bias
    biasrx1day = mrx1day - orx1day
    biasrx5day = mrx5day - orx5day
    biasr10mm = mr10mm - or10mm
    biascdd = mcdd - ocdd

    # calculate the centralised RMS
    crms = ((1/nt)*np.nansum((odata-np.nanmean(odata,axis=0))**2,axis=0))**(1/2)
    
    # calculate the relative bias error
    ebiasrx1day = abs(biasrx1day)/crms
    ebiasrx5day = abs(biasrx5day)/crms
    ebiasr10mm = abs(biasr10mm)/crms
    ebiascdd = abs(biascdd)/crms

    # calculate the bias score
    sbiasrx1day = np.exp(-ebiasrx1day)
    sbiasrx5day = np.exp(-ebiasrx5day)
    sbiasr10mm = np.exp(-ebiasr10mm)
    sbiascdd = np.exp(-ebiascdd)

    if flag == True:
        # calculate the area-weighted average
        latr = np.deg2rad(lat2d)
        weights = np.cos(latr)
        sbiasrx1day_wgt = np.ma.average(np.ma.MaskedArray(sbiasrx1day, mask=np.isnan(sbiasrx1day)),weights=weights)
        sbiasrx5day_wgt = np.ma.average(np.ma.MaskedArray(sbiasrx5day, mask=np.isnan(sbiasrx5day)),weights=weights)
        sbiasr10mm_wgt = np.ma.average(np.ma.MaskedArray(sbiasr10mm, mask=np.isnan(sbiasr10mm)),weights=weights)
        sbiascdd_wgt = np.ma.average(np.ma.MaskedArray(sbiascdd, mask=np.isnan(sbiascdd)),weights=weights)

        return [sbiasrx1day_wgt,sbiasrx5day_wgt,sbiasr10mm_wgt,sbiascdd_wgt]

    else:
        
        sbias = np.empty((4,nx,ny),dtype=np.float64)
        sbias[0,:,:] = sbiasrx1day
        sbias[1,:,:] = sbiasrx5day
        sbias[2,:,:] = sbiasr10mm
        sbias[3,:,:] = sbiascdd
        return sbias

# Calculates the spatial distribution score
def calc_spatialdsn_pr(mdata,odata,lat2d):
    """This function calculates the spatial distribution score using 3D [time,lat,lon] data
    Inputs:
        mdata == the model data
        odata == the observational data
        lat2d == the latitudes to calculate the weighted area average
        tx == calculate TX or TN extremes"""

    nt = mdata.shape[0]
    nx = mdata.shape[1]
    ny = mdata.shape[2]    
    nxy = mdata.shape[1]*mdata.shape[2]

    mdata = np.ma.masked_array(mdata, mdata>=1.e20).filled(np.nan)
    odata = np.ma.masked_array(odata, odata>=1.e20).filled(np.nan)
        
    # calculate the extremes 

    # RX1DAY
    mrx1day = np.nanmax(mdata,axis=0)
    orx1day = np.nanmax(odata,axis=0)
    
    # RX5DAY
    mrx5day = np.nanmax(ndimage.uniform_filter(mdata, size=(5,0,0)),axis=0)
    orx5day = np.nanmax(ndimage.uniform_filter(odata, size=(5,0,0)),axis=0)
    
    # CDD
    mcdd = np.empty((nx,ny),dtype=np.float64)
    ocdd = np.empty((nx,ny),dtype=np.float64)
    mrain = np.where(mdata < 1., 1, 0) # set all days with rain < 1mm to 1 and 0 otherwise
    orain = np.where(odata < 1., 1, 0)
    
    for ii in range(nx):
        for jj in range(ny):
                   
            #https://stackoverflow.com/questions/22214086/python-a-program-to-find-the-length-of-the-longest-run-in-a-given-list
            mcdd[ii,jj] = max(sum(1 for _ in l) for n, l in itertools.groupby(mrain[:,ii,jj]))
            ocdd[ii,jj] = max(sum(1 for _ in l) for n, l in itertools.groupby(orain[:,ii,jj]))
    
    # R10mm - number of days with >10mm
    mr10mm = np.nansum(np.where(mdata >= 10., 1, 0),axis=0)
    or10mm = np.nansum(np.where(odata >= 10., 1, 0),axis=0)  
            
    # calculate the normalised standard deviation
    #np.ma.MaskedArray(, mask=np.isnan())
    nstdrx1day = np.nanstd(mrx1day) / np.nanstd(orx1day)
    nstdrx5day = np.nanstd(mrx5day) / np.nanstd(orx5day)
    nstdr10mm = np.nanstd(mr10mm) / np.nanstd(or10mm)
    nstdcdd = np.nanstd(mcdd) / np.nanstd(ocdd)
    
    # calculate the spatial correlation coefficient
    rrx1day = ((1/nxy)*np.nansum((mrx1day-np.nanmean(mrx1day))*(orx1day-np.nanmean(orx1day))))/(np.nanstd(mrx1day)*np.nanstd(orx1day))
    rrx5day = ((1/nxy)*np.nansum((mrx5day-np.nanmean(mrx5day))*(orx5day-np.nanmean(orx5day))))/(np.nanstd(mrx5day)*np.nanstd(orx5day))
    rr10mm = ((1/nxy)*np.nansum((mr10mm-np.nanmean(mr10mm))*(or10mm-np.nanmean(or10mm))))/(np.nanstd(mr10mm)*np.nanstd(or10mm))
    rcdd = ((1/nxy)*np.nansum((mcdd-np.nanmean(mcdd))*(ocdd-np.nanmean(ocdd))))/(np.nanstd(mcdd)*np.nanstd(ocdd))
    
    # calculate the spatial distribution score
    sdistrx1day = (2*(1+rrx1day))/((nstdrx1day + (1/nstdrx1day))**2)
    sdistrx5day = (2*(1+rrx5day))/((nstdrx5day + (1/nstdrx5day))**2)
    sdistr10mm = (2*(1+rr10mm))/((nstdr10mm + (1/nstdr10mm))**2)
    sdistcdd = (2*(1+rcdd))/((nstdcdd + (1/nstdcdd))**2)
        
    return [sdistrx1day,sdistrx5day,sdistr10mm,sdistcdd]

