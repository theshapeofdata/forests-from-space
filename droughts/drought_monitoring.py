#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drought plots
Created on Sun Aug 25 22:08:53 2019

@author: ashwinip
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
import cartopy
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime as dt
from datetime import datetime as ts

#%% Read files

print('Reading files...')

# GPIs (Vegetatioon parameters)
"""
No. of grid points: 63865
Amazon_gpis_all.csv file contains all GPIs (and respective coordinates) in the study area.
"""
fname_csv_gpis=os.path.join('/Users','ashwinip','Documents','Academic','Thesis',
                        'Library Amazonia','Data','Big data',
                        'Amazon_gpis_all.csv')

gpis_df=pd.read_csv(fname_csv_gpis,sep=';') #All GPIs in study area
no_gpis = len(gpis_df) #total no. of GPIs in study area

# Backscatter - Descending Overpass
"""
Data length: 2007-16
No. of data points/grid point: 360
"""
fname_netcdf = os.path.join('/Users','ashwinip','Documents','Academic','Thesis',
                            'Library Amazonia','Data','Big data',
                            'Data_Amazon_sig_desc.nc')

with Dataset(fname_netcdf) as nc:
    DOY = nc.variables['jd_sigd'][:]
    #sigma_d = nc.variables['SIGMA_D'][:][:]

ref_jd = pd.Timestamp(0).to_julian_date()
datestampB=(pd.to_datetime(DOY-ref_jd,unit='D')).tolist()

    # Sigma20,40,60 curves
"""
sigma_angles.nc contains sigma20,40,60 climatologies and drought year values.
"""
fname_netcdf = os.path.join('/Users','ashwinip','Documents','Academic','Thesis',
                            'Library Amazonia','Data','Big data',
                            'sigma_theta.nc')

with Dataset(fname_netcdf) as nc:
    doy_b = nc.variables['doy_b'][:] #DOY for backscatter
    
    clim_sigma60_d = nc.variables['clim_sigma60_d'][:][:] #sigma60 climatology (desc.)
    clim_sigma60_a = nc.variables['clim_sigma60_a'][:][:] #sigma60 climatology (asc.)
    
    sigma60_10d = nc.variables['sigma60_10d'][:][:] #sigma60 in 2010 (desc.)
    sigma60_10a = nc.variables['sigma60_10a'][:][:] #sigma60 in 2010 (asc.)
    
    sigma60_15d = nc.variables['sigma60_15d'][:][:] #sigma60 in 2015 (desc.)
    sigma60_15a = nc.variables['sigma60_15a'][:][:] #sigma60 in 2015 (asc.)
    
    std_sigma60_d = nc.variables['std_sigma60_d'][:][:]#standard deviation of sigma60 (desc.)
    std_sigma60_a = nc.variables['std_sigma60_a'][:][:]#standard deviation of sigma60 (asc.)
    

doy_b = np.array(doy_b)

clim_sigma60_d=pd.DataFrame(clim_sigma60_d,index=doy_b)  #Dataframe for climatology of sigma20 (desc.)
clim_sigma60_a=pd.DataFrame(clim_sigma60_a,index=doy_b)  #Dataframe for climatology of sigma20 (asc.)

sigma60_10d=pd.DataFrame(sigma60_10d,index=doy_b)  #Dataframe for sigma60 in 2010 (desc.)
sigma60_10a=pd.DataFrame(sigma60_10a,index=doy_b)  #Dataframe for sigma60 in 2010 (asc.)

sigma60_15d=pd.DataFrame(sigma60_15d,index=doy_b)  #Dataframe for sigma60 in 2015 (desc.)
sigma60_15a=pd.DataFrame(sigma60_15a,index=doy_b)  #Dataframe for sigma60 in 2015 (asc.)

std_sigma60_d=pd.DataFrame(std_sigma60_d,index=doy_b)  #Dataframe for standard deviation of sigma20 (desc.)
std_sigma60_a=pd.DataFrame(std_sigma60_a,index=doy_b)  #Dataframe for standard deviation of sigma20 (asc.)


# Region-wise indices
"""
xlsx file containing region-wise GPIs for vegetation parameters 
9 regions
"""
no_regions = 10

regions = {
            0 : 'Madeira-Tapajos moist forests (fSE)',
            1 : 'Napo moist forests (fNW)',
            2 : 'Guianan moist forests (fNE)',
            3 : 'Southwest Amazon moist forests (fSW)',
            4 : 'Tapajos-Xingu moist forests (fSE)',
            5 : 'Jurua-Purus moist forests (fC)',
            6 : 'Marajo varzea (ff)',
            7 : 'Cerrado (sC)',
            8 : 'Guianan savanna (sG)',
            9 : 'Beni savanna (sB)'   
                }

    # vegetation parameters
fname_regions=os.path.join('/Users','ashwinip','Documents','Academic','Thesis',
                        'Python','csv',
                        'Indices_Ecoregions_Filtered.xlsx')

regions_df=pd.read_excel(fname_regions, sheet_name='Sheet1') #Reading Excel file

no_gpis_regions = np.zeros((no_regions))#No. of points in regions
index_region = []

for i in range(0,no_regions):
    no_gpis_regions[i] = len(regions_df.iloc[:,i].dropna())
    #index_region.append(gpis_df[np.isin(gpis_df['gpi'],regions_df.iloc[:,i].dropna())].index.values)
    index_region.append(regions_df.iloc[:,i].dropna().values)


print('Files read.')

#%% Region-wise means

std_sigma60mean_d = np.zeros((len(doy_b),no_regions))
std_sigma60mean_a = np.zeros((len(doy_b),no_regions))

sigma60_climmean_d = np.zeros((len(doy_b),no_regions))
sigma60_climmean_a = np.zeros((len(doy_b),no_regions))

sigma60_mean_10d = np.zeros((len(doy_b),no_regions))
sigma60_mean_10a = np.zeros((len(doy_b),no_regions))

sigma60_mean_15d = np.zeros((len(doy_b),no_regions))
sigma60_mean_15a = np.zeros((len(doy_b),no_regions))

for i in range(0,no_regions):

    sigma60_climmean_d[:,i] = np.mean(clim_sigma60_d.iloc[:,index_region[i]],axis=1)
    sigma60_climmean_a[:,i] = np.mean(clim_sigma60_a.iloc[:,index_region[i]],axis=1)
    
    sigma60_mean_10d[:,i] = np.mean(sigma60_10d.iloc[:,index_region[i]],axis=1)
    sigma60_mean_10a[:,i] = np.mean(sigma60_10a.iloc[:,index_region[i]],axis=1)
    
    sigma60_mean_15d[:,i] = np.mean(sigma60_15d.iloc[:,index_region[i]],axis=1)
    sigma60_mean_15a[:,i] = np.mean(sigma60_15a.iloc[:,index_region[i]],axis=1)
    
    std_sigma60mean_d[:,i] = np.mean(std_sigma60_d.iloc[:,index_region[i]],axis=1)
    std_sigma60mean_a[:,i] = np.mean(std_sigma60_a.iloc[:,index_region[i]],axis=1)
    
#%% Drought plots (2015 drought)

lgnd_droughts = [
                    '2015',
                    'Climatology',
                    '+/- 1 SD'
                    ] #legend for plots

for i in range(0,5):
     
        #Sigma40
    plt.figure(figsize=(10,5))
    plt.plot(doy_b,sigma60_mean_15d[:,i],'red',linewidth=2,zorder=10)
    plt.plot(doy_b,sigma60_climmean_d[:,i],'k',linewidth=1.5)
    plt.fill_between(doy_b, sigma60_climmean_d[:,i]+std_sigma60mean_d[:,i], sigma60_climmean_d[:,i]-std_sigma60mean_d[:,i], facecolor='grey', interpolate=True,alpha=0.3)
    
    title = regions[i] + '\n$\sigma^{\circ}_{60}$ (desc.): 2015 drought' #title
    plt.title(title,fontname='Times New Roman',fontsize=18)
    plt.ylabel('Backscatter [dB]',fontname='Times New Roman',fontsize=14)
    plt.xlabel('DOY',fontname='Times New Roman',fontsize=14)
    plt.xlim((min(doy_b),max(doy_b)))
    plt.legend(lgnd_droughts,fontsize=9)
    
    figname = 'B15R'+str(i)+'tsD'
    plt.savefig(figname)

#%% Diurnal differences (2010)

sigma60_climmean_dd = sigma60_climmean_d - sigma60_climmean_a

sigma60_mean_10dd = sigma60_mean_10d - sigma60_mean_10a

# Drought days
Dmin=153
Dmax=274
    
lgnd_droughts = [
                    '2010',
                    'Peak drought period',
                    'Climatology'
                    #'+/- 1 SD'
                    ] #legend for plots

for i in range(0,no_regions):
     
        #Sigma40
    plt.figure(figsize=(10,5))
    plt.plot(doy_b,sigma60_mean_10dd[:,i],'red',linewidth=2,zorder=10)
    plt.axvline(Dmin,linestyle='--',label='Peak drought period',zorder=10)
    plt.plot(doy_b,sigma60_climmean_dd[:,i],'k',linewidth=1.5)
    #plt.fill_between(doy_b, sigma60_climmean_d[:,i]+std_sigma60mean_d[:,i], sigma60_climmean_d[:,i]-std_sigma60mean_d[:,i], facecolor='grey', interpolate=True,alpha=0.3)
    plt.axvline(Dmax,linestyle='--',zorder=10)
    plt.axhline(0,linestyle='--',color='k',linewidth=0.8,zorder=1)
    
    title = regions[i+1] + '\n2010 drought: Diurnal differences in $\sigma^{\circ}_{60}$' #title
    plt.title(title,fontname='Times New Roman',fontsize=18)
    plt.ylabel('Backscatter [dB]',fontname='Times New Roman',fontsize=14)
    plt.xlabel('DOY',fontname='Times New Roman',fontsize=14)
    plt.xlim((min(doy_b),max(doy_b)))
    plt.legend(lgnd_droughts,fontsize=9)
    
    figname = 'B6010R'+str(i+1)+'tsdd'
    plt.savefig(figname, bbox_inches = 'tight')

#%% Diurnal differences (2015)

sigma60_climmean_dd = sigma60_climmean_d - sigma60_climmean_a

sigma60_mean_15dd = sigma60_mean_15d - sigma60_mean_15a

# Drought days
Dmin=275
Dmax=364

lgnd_droughts = [
                    '2015',
                    'Peak drought interval',
                    'Climatology'
                    #'+/- 1 SD'
                    ] #legend for plots

for i in range(0,no_regions):
     
        #Sigma40
    plt.figure(figsize=(10,5))
    plt.plot(doy_b,sigma60_mean_15dd[:,i],'red',linewidth=2,zorder=10)
    plt.axvline(Dmin,linestyle='--',label='Peak drought period',zorder=10)
    plt.plot(doy_b,sigma60_climmean_dd[:,i],'k',linewidth=1.5)
    #plt.fill_between(doy_b, sigma60_climmean_d[:,i]+std_sigma60mean_d[:,i], sigma60_climmean_d[:,i]-std_sigma60mean_d[:,i], facecolor='grey', interpolate=True,alpha=0.3)
    plt.axvline(Dmax,linestyle='--',zorder=10)
    plt.axhline(0,linestyle='--',color='k',linewidth=0.8,zorder=1)
    
    title = regions[i+1] + '\n2015 drought: Diurnal differences in $\sigma^{\circ}_{60}$' #title
    plt.title(title,fontname='Times New Roman',fontsize=18)
    plt.ylabel('Backscatter [dB]',fontname='Times New Roman',fontsize=14)
    plt.xlabel('DOY',fontname='Times New Roman',fontsize=14)
    plt.xlim((min(doy_b),max(doy_b)))
    plt.legend(lgnd_droughts,fontsize=9)
    
    figname = 'B6015R'+str(i+1)+'tsdd'
    plt.savefig(figname, bbox_inches = 'tight')

