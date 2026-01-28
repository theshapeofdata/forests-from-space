#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots to elucidate flooding effect.
Created on Thu Jul 25 16:04:52 2019

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
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

#%% Reading files

# Climatologies of vegetation parameters

"""
Data length: 2007-16
climatologies.nc is a netCDF file containing climatologies of all parameters (both overpasses).
"""
fname_netcdf = os.path.join('/Users','ashwinip','Documents','Academic','Thesis',
                            'Library Amazonia','Data','Big data',
                            'climatologies.nc')

with Dataset(fname_netcdf) as nc:
    doy = nc.variables['DOY'][:] #Day of year (for slope and curvature)
    doy_b = nc.variables['DOY_B'][:] #Day of year (for backscatter)
    clim_slope_d = nc.variables['clim_slope_d'][:][:] #slope (desc.)
    clim_curve_d = nc.variables['clim_curve_d'][:][:] #curvature (desc.)
    clim_sigma_d = nc.variables['clim_sigma_d'][:][:] #backscatter (desc.)
    clim_slope_a = nc.variables['clim_slope_a'][:][:] #slope (asc.)
    clim_curve_a = nc.variables['clim_curve_a'][:][:] #curvature (asc.)
    clim_sigma_a = nc.variables['clim_sigma_a'][:][:] #backscatter (asc.)

doy = np.array(doy)
doy_b = np.array(doy_b)

index_B=np.isin(doy,doy_b) #doys common to all parameters

clim_slope_d=pd.DataFrame(clim_slope_d,index=doy)  #Dataframe for climatology of slope (desc.)
clim_curve_d=pd.DataFrame(clim_curve_d,index=doy)  #Dataframe for climatology of curvature (desc.)
clim_sigma_d=pd.DataFrame(clim_sigma_d,index=doy_b)  #Dataframe for climatology of sigma40 (desc.)
clim_slope_a=pd.DataFrame(clim_slope_a,index=doy)  #Dataframe for climatology of slope (asc.)
clim_curve_a=pd.DataFrame(clim_curve_a,index=doy)  #Dataframe for climatology of curvature (asc.)
clim_sigma_a=pd.DataFrame(clim_sigma_a,index=doy_b)  #Dataframe for climatology of sigma40 (asc.)

# Region-wise indices
"""
xlsx file containing region-wise GPIs for precipitation data
9 regions
"""
regions = {
            1 : 'Napo moist forests (NW)',
            2 : 'Guianan moist forests (NE)',
            3 : 'Southwest Amazon moist forests (SW)',
            4 : 'Tapajos-Xingu moist forests (SE)',
            5 : 'Jurua-Purus moist forests (Central)',
            6 : 'Marajo varzea',
            7 : 'Cerrado',
            8 : 'Guianan savanna',
            9 : 'Beni savanna'   
                }
no_regions = 9

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

#%% Backscatter curve (flooded and non-flooded)

# Marajo varzea

region = 5

    #Calculate mean climatologies
meanslope = np.mean(clim_slope_d.iloc[index_B,index_region[region]],axis=1)
meancurv = np.mean(clim_curve_d.iloc[index_B,index_region[region]],axis=1)
meansig40 = np.mean(clim_sigma_d.iloc[:,index_region[region]],axis=1)

    #Generate backscatter curves
thetas = np.arange(20.0, 70.0) #incidence angle
sigma = np.zeros((len(doy_b), len(thetas))) #backscatter

for dd, ddoy in enumerate(doy_b):
    for tt, theta in enumerate(thetas):
        sigma[dd, tt] = meansig40.iloc[dd] + \
                meanslope.iloc[dd] * (theta - 40) + \
                0.5 * meancurv.iloc[dd] * ((theta - 40) ** 2)

plt.figure(figsize=(9,6))
plt.plot(thetas,sigma[10,:],label='Flooded (DOY 109)',color='dodgerblue')
plt.plot(thetas,sigma[29,:],label='Not flooded (DOY 303)',color='darkgoldenrod')
plt.title('Marajo varzea\n$\sigma-\Theta$ curve on flooded and non-flooded days',
          fontname='Times New Roman',fontsize=22)
plt.ylabel('Backscatter [dB]',fontname='Times New Roman',fontsize=16)
plt.xlabel('$\Theta$ [deg]',fontname='Times New Roman',fontsize=16)
plt.legend(fontsize=12)
plt.savefig('BTfloodMarajo',dpi=300)

# Beni savanna

region = 8

    #Calculate mean climatologies
meanslope = np.mean(clim_slope_d.iloc[index_B,index_region[region]],axis=1)
meancurv = np.mean(clim_curve_d.iloc[index_B,index_region[region]],axis=1)
meansig40 = np.mean(clim_sigma_d.iloc[:,index_region[region]],axis=1)

    #Generate backscatter curves
thetas = np.arange(20.0, 70.0) #incidence angle
sigma = np.zeros((len(doy_b), len(thetas))) #backscatter

for dd, ddoy in enumerate(doy_b):
    for tt, theta in enumerate(thetas):
        sigma[dd, tt] = meansig40.iloc[dd] + \
                meanslope.iloc[dd] * (theta - 40) + \
                0.5 * meancurv.iloc[dd] * ((theta - 40) ** 2)
plt.figure(figsize=(9,6))
plt.plot(thetas,sigma[6,:],label='Flooded (DOY 68)',color='dodgerblue')
plt.plot(thetas,sigma[29,:],label='Not flooded (DOY 303)',color='darkgoldenrod')
plt.title('Beni savanna\n$\sigma-\Theta$ curve on flooded and non-flooded days',
          fontname='Times New Roman',fontsize=22)
plt.ylabel('Backscatter [dB]',fontname='Times New Roman',fontsize=16)
plt.xlabel('$\Theta$ [deg]',fontname='Times New Roman',fontsize=16)
plt.legend(fontsize=12)
plt.savefig('BTfloodBeni',dpi=300)


#%% Climatologies

# Marajo varzea

region = 5

    #Calculate mean climatologies
meanslope = np.mean(clim_slope_d.iloc[index_B,index_region[region]],axis=1)
meancurv = np.mean(clim_curve_d.iloc[index_B,index_region[region]],axis=1)
meansig40 = np.mean(clim_sigma_d.iloc[:,index_region[region]],axis=1)

    #Generate backscatter curves
thetas = np.arange(20.0, 70.0) #incidence angle
sigma = np.zeros((len(doy_b), len(thetas))) #backscatter

for dd, ddoy in enumerate(doy_b):
    for tt, theta in enumerate(thetas):
        sigma[dd, tt] = meansig40.iloc[dd] + \
                meanslope.iloc[dd] * (theta - 40) + \
                0.5 * meancurv.iloc[dd] * ((theta - 40) ** 2)
                

plt.figure(figsize=(9,6))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 0
new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par1,
                                    offset=(offset, 0))

par1.axis["right"].toggle(all=True)

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)

host.set_xlabel("DOY",fontsize=20)
host.set_ylabel("$\sigma_{20}$",fontsize=22)
par1.set_ylabel("$\sigma_{40}$",fontsize=22)
par2.set_ylabel("$\sigma_{60}$",fontsize=22)

p1, = host.plot(doy_b, sigma[:, 0], zorder=10, color='dodgerblue', label="$\sigma_{20}$", linewidth=1.7)
p2, = par1.plot(doy_b, sigma[:, 20], zorder=1, color='k', label="$\sigma_{40}$")
p3, = par2.plot(doy_b, sigma[:, 40], zorder=2, color='forestgreen', label="$\sigma_{60}$")

#par1.set_ylim(150, 200)
#par2.set_ylim(2,14)

host.legend(fontsize=12)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

title='Marajo varzea\nBackscatter climatology for different incidence angles'
host.set_title(title,fontsize=16,fontname='Times New Roman')

    #Save figure
figname='BTtsMarajo'
plt.savefig(figname,dpi=300)


# Beni savanna

region = 8

    #Calculate mean climatologies
meanslope = np.mean(clim_slope_d.iloc[index_B,index_region[region]],axis=1)
meancurv = np.mean(clim_curve_d.iloc[index_B,index_region[region]],axis=1)
meansig40 = np.mean(clim_sigma_d.iloc[:,index_region[region]],axis=1)

    #Generate backscatter curves
thetas = np.arange(20.0, 70.0) #incidence angle
sigma = np.zeros((len(doy_b), len(thetas))) #backscatter

for dd, ddoy in enumerate(doy_b):
    for tt, theta in enumerate(thetas):
        sigma[dd, tt] = meansig40.iloc[dd] + \
                meanslope.iloc[dd] * (theta - 40) + \
                0.5 * meancurv.iloc[dd] * ((theta - 40) ** 2)
                

plt.figure(figsize=(9,6))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 0
new_fixed_axis = par1.get_grid_helper().new_fixed_axis
par1.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par1,
                                    offset=(offset, 0))

par1.axis["right"].toggle(all=True)

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)

host.set_xlabel("DOY",fontsize=20)
host.set_ylabel("$\sigma_{20}$",fontsize=22)
par1.set_ylabel("$\sigma_{40}$",fontsize=22)
par2.set_ylabel("$\sigma_{60}$",fontsize=22)

p1, = host.plot(doy_b, sigma[:, 0], zorder=10, color='dodgerblue', label="$\sigma_{20}$", linewidth=1.7)
p2, = par1.plot(doy_b, sigma[:, 20], zorder=1, color='k', label="$\sigma_{40}$")
p3, = par2.plot(doy_b, sigma[:, 40], zorder=2, color='forestgreen', label="$\sigma_{60}$")

#par1.set_ylim(150, 200)
#par2.set_ylim(2,14)

host.legend(fontsize=12)

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

title='Beni savanna\nBackscatter climatology for different incidence angles'
host.set_title(title,fontsize=16,fontname='Times New Roman')

    #Save figure
figname='BTtsBeni'
plt.savefig(figname,dpi=300)
