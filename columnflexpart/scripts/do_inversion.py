   
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from columnflexpart.utils import select_boundary, optimal_lambda
from columnflexpart.classes.flexdataset import FlexDataset
from functools import partial
import geopandas as gpd
import cartopy.crs as ccrs
from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
import bayesinverse
from columnflexpart.classes.inversion import InversionBioclass
from columnflexpart.utils.utils import optimal_lambda



#####Plotting input mask ####
def plot_input_mask(savepath,datapath_and_name, selection= None): 
    ds = xr.open_dataset(datapath_and_name)
    if selection != None: 
        ds0 = ds.where(ds.bioclass==selection[0])
        for i in selection: 
            dsi = ds.where(ds.bioclass==i, drop = True)
            ds0 = xr.merge([ds0,dsi])
        ds = ds0
    #ds = ds.where((ds.bioclass==5)|(ds.bioclass==6)|(ds.bioclass==7)|(ds.bioclass==8)|(ds.bioclass==16)|(ds.bioclass==20)|(ds.bioclass==31))
    plt.figure(figsize=(14, 10))    
    plt.rcParams.update({'font.size':25})    
    ax = plt.axes(projection=ccrs.PlateCarree())     
    fig = ds['bioclass'].plot(x = 'Long', y = 'Lat',ax = ax ,cmap = 'nipy_spectral', cbar_kwargs = dict(label='region number', shrink = 0.88))#gist_ncar
    #fig.cbar.set_label(label='region number', size = 20)
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    if selection!=None: 
        plt.savefig(savepath+'selected_bioclasses.png')
    else:
        print('Region labels: '+str(set(ds.bioclass.values.flatten())))
        print('Number of regions: '+str(len(set(ds.bioclass.values.flatten()))))
        plt.savefig(savepath+'bioclasses.png')
    return len(set(ds.bioclass.values.flatten()))
    


### Plotting spatial prior ########
def plot_prior_spatially(Inversion,week_min, week_max, savepath):
    flux_mean, flux_err = Inversion.get_flux()
    #print(flux_mean[:,flux_mean['week']==week])
    plt.rcParams.update({'font.size':13})    
    for week in range(week_min, week_max+1):
        spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
        #print(flux_mean)
        spatial_flux = spatial_flux*12*10**6
        plt.figure()
        spatial_flux.plot(x = 'longitude', y = 'latitude',  cmap = 'seismic',vmin = -30, cbar_kwargs = {'label' : r'flux [$\mu$ gCm$^{-2}$s$^{-1}$]'})
        plt.title('Week '+str(week))
        plt.savefig(savepath+'CT_mean_flux_spatial_week'+str(week)+'.png')

def plot_l_curve(Inversion,savepath, alpha):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]
    inv_result = Inversion.compute_l_curve(alpha =[1e-8,3e-8, 1e-7, 4.92e-7, 1.11e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1.41e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1])
    plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    fig, ax = Inversion.plot_l_curve(mark_ind = 15, mark_kwargs= dict(color = 'firebrick',label = r'$\lambda = 1.41 \cdot 10^{-3}$'))
    plt.grid(axis = 'y', color = 'grey', linestyle = '--' )
    plt.legend()
    plt.savefig(savepath+'l_curve_'+str(alpha)+'.png')


#plot_l_curve(Inversion)
def plot_averaging_kernel(Inversion, alpha, class_num, week_num,savepath):
    flux_mean, flux_err = Inversion.get_flux()
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) 
    #result = Inversion.compute_l_curve(alpha = [0.004317500056754969], xerr = err)
    ak = Inversion.get_averaging_kernel()
  
    ak_sum = np.zeros((week_num*class_num,class_num))    
    for i in range(0,class_num):#ecosystem number 
        list_indices = list(i+class_num*np.arange(0,week_num))
        #print(list_indices)
        for idx in list_indices: 
            ak_sum[:,i] += ak[:,idx]#ak[:,i]+ak[:,i+7]+ak[:,i+2*7]+ak[:,i+3*7]+ak[:,i+4*7]+ak[:,i+5*7]
       
    ak_final = np.zeros((class_num,class_num))
    for i in range(0,class_num):
        list_indices = list(i+class_num*np.arange(0,week_num))#0,week_num))
        for idx in list_indices:
            ak_final[i] += ak_sum[idx]#(ak_sum[i+7*2]+ak_sum[i+7*3]+ak_sum[i+7*4]+ak_sum[i+7*5])/4
        ak_final[i] = ak_final[i]/len(list_indices)
    #ak_sum[i]+ak_sum[i+7]+
    #ak_sum = ak.sum(axis =1)#ak[0:7,:].dot(np.ones(7))
    #print(ak_final)
    #sk_shape # Seite 47
    plt.figure()
    plt.imshow(ak_final, vmax = 1, vmin = 0)
    plt.colorbar()
    plt.savefig(savepath+'ak_final_2d_'+str(alpha)+'.png')


#plot_averaging_kernel(Inversion, 1.1116017136941747e-06, 27, 6)


##### Versuch Gesamtemissionen nach inversion zu berechnen 
def calc_concentrations():
    flux_mean, flux_err = Inversion.get_flux()
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = 0.004317500056754969, xerr = err) 
    print(predictions)
    fp, conc, conc_err = Inversion.get_footprint_and_measurement('xco2')
    #print(result[0])
    result_spatial = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
    result_spatial.plot(x = 'longitude', y = 'latitude')
    plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup6/footprint_coarsened.png')

#for week in range(48,53):
    plot_prior_spatially(Inversion, week)
#


'''
print(err)
### Plotting error flux ####
spatial_flux = Inversion.map_on_grid(err[:,err['week']==week])
#print(flux_mean)
spatial_flux = spatial_flux*12*10**6
spatial_flux.plot(x = 'longitude', y = 'latitude',  cmap = 'seismic', cbar_kwargs = {'label' : r'flux [$\mu$gCm$^{-2}$s$^{-1}$]'})
plt.savefig('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/CT_mean_flux_err_spatial_week'+str(week)+'.png')


'''
def find_two_optimal_lambdas(Inversion, range, stop):
    l = optimal_lambda(Inversion,range, stop)
    print('Optimal lambda 1: '+str(l))
    l2 = optimal_lambda(Inversion,[l, range[1]], stop)
    print('Optimal lambda 2: '+str(l2))
    return l, l2


####### Find optimat lambda #####
#l = optimal_lambda(Inversion,[1e-5,1], 1e-9) # 4.920278906480012e-07 # 0.004317500056754969 wenn der davor ausgeschlossen wurde
# für setup6 : 1.1116017136941747e-06 or 0.0014135529660449899
# für version 5 : 5.50553658945752e-06 or 0.0013781916546518236
#print(l)

### print spatial result or diferences plot fluxes #########
def plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,alpha,vminv=None, diff =False):

    flux_mean, flux_err = Inversion.get_flux()
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) # was macht alpha? 
    for week in range(week_min,week_max+1): 
        #print(predictions)
        plt.figure()
        spatial_result = Inversion.map_on_grid(predictions[predictions['week']==week])
        if diff== True: 
            spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
            spatial_flux = spatial_flux *12*10**6
            spatial_result = (spatial_result*12*10**(6))-spatial_flux
            spatial_result.plot(x = 'longitude', y = 'latitude', cmap = 'seismic',vmin = vminv,cbar_kwargs = {'label' : r'flux [$\mu$gC m$^{-2}$ s$^{-1}$]'})
            plt.title('Week '+str(week))
            plt.savefig(savepath+'Diff_to_prior_week_'+str(week)+'_alpha_xerr.png')
        else: 

            spatial_result = (spatial_result*12*10**(6))
            spatial_result.plot(x = 'longitude', y = 'latitude', cmap = 'seismic',vmin = vminv,cbar_kwargs = {'label' : r'flux [$\mu$gC m$^{-2}$ s$^{-1}$]'})
            plt.title('Week '+str(week))
            plt.savefig(savepath+'Spatial_results_week_'+str(week)+'_alpha'+str(alpha)+'_xerr.png')

    #print(predictions)
    #get_total(Inversion)



def do_everything(savepath, Inversion, mask_datapath_and_name, week_min, week_max, week_num):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    l1, l2 = find_two_optimal_lambdas(Inversion,[1e-8,1], 1e-9)
    for l in [l1, l2]: 
        #plot_l_curve(Inversion,savepath, l)
        plot_prior_spatially(Inversion,week_min, week_max, savepath)
        print('Plotting spatial results')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,l,vminv=None, diff =False)
        print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,l,vminv=None, diff =True)
        print('Plotting averaging kernels')
        plot_averaging_kernel(Inversion, l, class_num, week_num,savepath)



savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_version7/'

Inversion = InversionBioclass(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path= "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania",
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)


do_everything(savepath, Inversion,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania",
              1,1,6)
do_everything(savepath, Inversion,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania",
              48,52,6)
#class_num = plot_input_mask(savepath,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania")#,selection = [5,6,7,8,16,20,31])
