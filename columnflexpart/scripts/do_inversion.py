   
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
from columnflexpart.classes.inversionCO import InversionBioclassCO
#from columnflexpart.utils.utilsCO import optimal_lambda
from matplotlib.colors import LogNorm 
from matplotlib.dates import DateFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


###for centering diverging coloarbar at white:
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def add_iso_line(ax, x, y, z, value, **kwargs):
    v = np.diff(z > value, axis=1)
    h = np.diff(z > value, axis=0)
    l = np.argwhere(v.T)
    vlines = np.array(list(zip(np.stack((x[l[:, 0]+1], y[l[:, 1] ])).T,
    np.stack((x[l[:, 0] + 1], y[l[:, 1] + 1])).T)))
    l = np.argwhere(h.T)
    hlines = np.array(list(zip(np.stack((x[l[:, 0]], y[l[:, 1] + 1])).T,
    np.stack((x[l[:, 0] + 1], y[l[:, 1] + 1])).T)))
    lines = np.vstack((vlines, hlines))
    #colors = [mcolors.to_rgba(c)
    #      for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    ax.add_collection(LineCollection(lines, **kwargs))


#####Plotting input mask ####
def plot_input_mask(savepath,datapath_and_name, selection= None): 
    ds = xr.open_dataset(datapath_and_name)
    if selection != None: 
        ds0 = ds.where(ds.bioclass==selection[0])
        for i in selection: 
            dsi = ds.where(ds.bioclass==i, drop = True)
            ds0 = xr.merge([ds0,dsi])
        ds = ds0
    plt.figure(figsize=(14, 10))    
    plt.rcParams.update({'font.size':25})    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    mask = ds 
    #dsp = ds.where(ds.bioclass.values > 0)
    colors = ['lightblue', 'seagreen', 'chocolate','darkseagreen', 'mediumseagreen','forestgreen', 'darkgreen']
    cmap = LinearSegmentedColormap.from_list("", colors)
    ds['bioclass'].plot(x = 'Long', y = 'Lat',add_colorbar = False, cmap = cmap)
    print(set(list(ds.bioclass.values.flatten())))
    for b in list(set(list(ds.bioclass.values.flatten())))[:-1]: 
        add_iso_line(ax = ax, x = mask.Long.values-0.5, y = mask.Lat.values-0.5, z = mask.bioclass.values, value = b+0.5, color="black", linewidth=0.2)
    #fig = ds['bioclass'].plot.contour(x = 'Long', y = 'Lat',cmap = 'nipy_spectral', row = ['x'], col = ['y'])#,ax = ax ,cmap = 'nipy_spectral', cbar_kwargs = dict(label='region number', shrink = 0.88))#gist_ncar
    #fig.cbar.set_label(label='region number', size = 20)
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    if selection!=None: 
        plt.savefig(savepath+'less_selected_bioclasses.png')
    else:
        print('Region labels: '+str(set(ds.bioclass.values.flatten())))
        print('Number of regions: '+str(len(set(ds.bioclass.values.flatten()))))
        plt.savefig(savepath+'bioclasses.png')
    return len(set(ds.bioclass.values.flatten()))
    


### Plotting spatial prior ########
def plot_prior_spatially(Inversion, molecule_name, week_min, week_max, savepath):
    plt.rcParams.update({'font.size':15})    
    factor = 10**6*12 
    flux_mean = Inversion.flux*factor

    for week in range(week_min, week_max+1):
        spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
        #spatial_flux.to_netcdf(savepath+'prior_spatially_week_'+str(week))
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  
        spatial_flux.plot(x = 'longitude', y = 'latitude',  cmap = 'seismic', ax = ax, vmax = 0.4, vmin = -0.4, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835})
        plt.scatter(x = 150.8793,y =-34.4061,color="black")
        ax.coastlines()
        #plt.title('Week '+str(week))
        plt.savefig(savepath+'Prior_'+molecule_name+'_mean_flux_spatial_week'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)

def plot_l_curve(Inversion,err, molecule_name, savepath, alpha):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]

    #1e-60,1e-55,1e-52,1e-50,1e-45,1e-40,1e-38, 1e-36, 1e-35,1e-34,1e-33,1e-32,1e-31, 5e-31, 2e-30,1e-29,5e-29, 1e-28,5e-28,1e-27,1e-26,5e-26, 1e-25,5e-25,1e-24, 1e-23,1e-22,1e-21, 1e-20,1e-19,1e-18, 1e-17,
    print('compute l curve')
    inv_result = Inversion.compute_l_curve(cond = 1e-14,alpha =[1e-5,4e-5, 1e-4,4e-4,1e-3,4e-3,1e-2,4e-2, 1e-1,4e-1, 1,4, 10], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [3e-19,1e-18,3e-18,1e-17,3e-17,1e-16,3e-16,1e-15,3e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 1e-11,3e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,
    #                                                5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [1e-17,5e-17,1e-16,5e-16,1e-15,5e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 5e-12,1e-11,5e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    print('Plotting')
    plt.figure()
    print(inv_result)
    #plt.scatter(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    #plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    plt.scatter(inv_result["loss_forward_model"],inv_result["loss_regularization"], color = 'black')
    plt.scatter(inv_result["loss_forward_model"][8], inv_result["loss_regularization"][8], color = 'firebrick', label = '$\lambda = 10^{-1}$')
    plt.plot(inv_result["loss_forward_model"],inv_result["loss_regularization"], color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Regularization loss')
    plt.xlabel('Forward model loss')
    #plt.xlim((0,1e7))
    #plt.ylim((0, 4e-13))
    #fig, ax = Inversion.plot_l_curve(mark_ind = 17, mark_kwargs= dict(color = 'firebrick',label = r'$\lambda =1 \cdot 10^{-3}$'))
    #plt.grid(axis = 'y', color = 'grey', linestyle = '--' )
    plt.legend()
    print('saving')
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_l_curve_xerr_extended2.png')


def plot_averaging_kernel(Inversion, molecule_name, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    #result = Inversion.compute_l_curve(alpha = [0.004317500056754969], xerr = err)
    ak = Inversion.get_averaging_kernel()
    plt.rcParams.update({'font.size':15})
    ak_sum = np.zeros((week_num*class_num,class_num))   
    for i in range(0,class_num):#ecosystem number 
        list_indices = list(i+class_num*np.arange(0,week_num))
        for idx in list_indices: 
            ak_sum[:,i] += ak[:,idx]#ak[:,i]+ak[:,i+7]+ak[:,i+2*7]+ak[:,i+3*7]+ak[:,i+4*7]+ak[:,i+5*7]
       
    ak_final = np.zeros((class_num,class_num))
    for i in range(0,class_num):
        list_indices = list(i+class_num*np.arange(0,week_num))#0,week_num))
        for idx in list_indices:
            ak_final[i] += ak_sum[idx]#(ak_sum[i+7*2]+ak_sum[i+7*3]+ak_sum[i+7*4]+ak_sum[i+7*5])/4
        ak_final[i] = ak_final[i]/len(list_indices)
    
    if plot_spatially ==  True and weekly == False: 
        ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  
        orig_map=plt.cm.get_cmap('gist_heat')
        reversed_map = orig_map.reversed()
        ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax,cmap = reversed_map,cbar_kwargs = {'shrink':0.835})
        plt.scatter(x = 150.8793,y =-34.4061,color="black")
        ax.coastlines()
        plt.title('CO')
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_ak_spatial_gist_heat_reversed'+'log_1e-3_no_ocean.png', bbox_inches = 'tight')


    #sk_shape # Seite 47
    elif plot_spatially == True and weekly == True: 
        week_list = [1,48,49,50,51,52]
        for week in range(0,week_num): 
            ak_xr = xr.DataArray(data = ak_sum[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            #ak_xr = select_region_bioclass_based(ak_xr)
            ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
            #ak_spatial.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_CO_spatially_week_'+str(week_list[week])+'.nc')
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())  
            orig_map=plt.cm.get_cmap('gist_heat')
            reversed_map = orig_map.reversed()
            ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax, vmax = 1, cmap = reversed_map, cbar_kwargs = {'shrink':0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('CO')# week '+str(week_list[week])+' 2019 ')
            fig.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_spatial_gist_heat_reversed_week'+str(week_list[week])+'_no_ocean.png', bbox_inches = 'tight', dpi = 450)
        #ak_xr = xr.DataArray(data = ak_sum[class_num:2*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[2*class_num:3*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[0:class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
    else: 
        plt.figure()
        plt.imshow(ak_final, vmin = 0, vmax = 1) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_final_2d.png')
 

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
def plot_spatial_flux_results_or_diff_to_prior(savepath,  Inversion,molecule_name, week_min, week_max,alpha,vminv=None, diff =False):
    factor = 12*10**6
    #total_spatial_result = xr.Dataset()
    print(alpha)
    plt.rcParams.update({'font.size':15})   
    for week in range(week_min,week_max+1): 
        plt.figure()
        #for bioclass in [0,1,2,3,4,5,6,7,8,10]: 
        #       #print(predictions)
        #       pred = predictions[predictions['week']==week]
        #       #print(predictions)
        #       spat_result_class = Inversion.map_on_grid(pred[:,pred['bioclass']==bioclass])
        #       spat_result_class.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_spatial_results_week_'+str(week)+'_bioclass_'+str(bioclass)+'.nc')
        #prediction = select_region_bioclass_based(Inversion.predictions[Inversion.predictions['week']==week])
        #spatial_result = Inversion.map_on_grid(prediction[prediction['week']==week])
        spatial_result = Inversion.map_on_grid(Inversion.predictions[Inversion.predictions['week']==week])
        ax = plt.axes(projection=ccrs.PlateCarree())  
        if diff== True: 
            spatial_flux = Inversion.map_on_grid(Inversion.flux[:,Inversion.flux['week']==week])
            spatial_flux = spatial_flux *factor
            spatial_result = (spatial_result*factor)-spatial_flux
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -10, vmax = 10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_Diff_to_prior_week_'+str(week)+'_xerr.png', bbox_inches = 'tight')
        else: 
            spatial_result = (spatial_result*factor)
            #spatial_result = spatial_result.where(spatial_result.values>0.5)
            print(week)
            print(spatial_result.min())
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -10, vmax = 10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_err_per_reg_Spatial_results_week_'+str(week)+'_xerr.png', bbox_inches = 'tight', dpi = 450)
            #spatial_result.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_spatial_results_week_'+str(week)+'.nc')
    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')


def calc_emission_factors(datapathCO, datapathCO2, alpha, week_min, week_max): 
    for week in range(week_min, week_max+1): 
        ds_co = xr.open_dataset(datapathCO+str("{:e}".format(alpha))+'_CO_spatial_results_week_'+str(week)+'nc')
        ds_co2 = xr.open_dataset(datapathCO2+str("{:e}".format(alpha))+'_CO2_spatial_results_week_'+str(week)+'.nc')

##### Versuch Gesamtemissionen nach inversion zu berechnen 
def calc_concentrations(Inversion, molecule_name,alpha, savepath):
    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    if molecule_name == 'CO':
        ds = pd.read_pickle(datapath_predictions+'predictions3_CO.pkl')
        factor = 10**9
    elif molecule_name =='CO2': 
        ds = pd.read_pickle(datapath_predictions+'predictions.pkl')
        factor = 10**6  

    concentration_results = Inversion.footprints_flat.values*Inversion.predictions_flat.values*factor
    conc_sum = concentration_results.sum(axis = 1)
    conc_tot = conc_sum + ds['background_inter']

    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (12,8))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])
    print(df.time.values)
   
    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
    ax.legend()
    ax.set_xticks([datetime(year =2019, month = 12, day = 1),datetime(year =2019, month = 12, day = 5), datetime(year =2019, month = 12, day = 10), datetime(year =2019, month = 12, day = 15),
                datetime(year =2019, month = 12, day = 20), datetime(year =2019, month = 12, day = 25), datetime(year =2019, month = 12, day = 30), 
                ], 
                rotation=45)#datetime(year = 2020, month = 1, day = 4)
    ax.set_xlim(left =  datetime(year = 2019, month = 11, day=30), right = datetime(year = 2019, month = 12, day=31, hour = 15))
    ax.grid(axis = 'both')
    ax.set_ylabel('concentration [ppm]', labelpad=6)
    ax.errorbar(x= datetime(year =2019, month = 12, day = 31, hour = 4), y = 35, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('date')

    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)

    ## Rotate date labels automatically
    fig.autofmt_xdate()
    #ax.legend()
    #ax.grid(axis = 'both')
    #ax.set_ylabel('concentration [ppm]')
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    ds_mean = ds_mean[(ds_mean['datetime']>=datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime(year=2020, month =1, day=9))]
    ax2.bar(ds_mean['datetime'], ds_mean['number_of_measurements'], width=0.1, color = 'dimgrey')
    ax2.set_ylabel('# measurements', labelpad=17)
    ax2.grid(axis='x')
    #ax.set_title('CO', fontsize = 30)
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ylabel_dist_concentrations_results_only_12.png', dpi = 300, bbox_inches = 'tight')

    return

    flux_mean, flux_err = Inversion.__xarray_dataarray_variable__
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) 
   #print(predictions)
    fp, conc, conc_err = Inversion.get_footprint_and_measurement('xco2')
    print(fp)
    #print(result[0])
    #result_spatial = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
    #result_spatial.plot(x = 'longitude', y = 'latitude')
    spatial_fp = Inversion.map_on_grid(fp[:,fp['measurement']==1,fp['week']==week])
    print(spatial_fp)
    print(spatial_fp['prior_date'])
    spatial_fp.plot(x='longitude', y='latitude')
    plt.savefig(savepath+'footprint_coarsened.png')



def plot_weekly_concentrations(Inversion, molecule_name,alpha, savepath):

    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    if molecule_name == 'CO':
        ds = pd.read_pickle(datapath_predictions+'predictions3_CO.pkl')
        unit = 'ppb'
        factor = 10**9
    elif molecule_name =='CO2': 
        ds = pd.read_pickle(datapath_predictions+'predictions.pkl')  
        unit = 'ppm'
        factor = 10**6
    
    concentration_results = Inversion.footprints_flat.values*Inversion.predictions_flat.values*factor
    conc_sum = concentration_results.sum(axis = 1)
    #print(conc_sum)
    conc_tot = conc_sum + ds['background_inter']
    plt.rcParams.update({'font.size':14})   
    plt.rcParams.update({'errorbar.capsize': 5})
    #fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (14,10))
    fig, ax = plt.subplots(1,1, figsize = (14,10))#plt.figure(figsize = (14,10))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])

    df48 = df[df['time']<datetime(year=2019, month = 12, day=2)]
    df49 = df[(df['time']>datetime(year=2019, month = 12, day=1, hour = 23))&(df['time']<datetime(year=2019, month = 12, day=9))]
    df50 = df[(df['time']>datetime(year=2019, month = 12, day=8, hour = 23))&(df['time']<datetime(year=2019, month = 12, day=16))]
    df51 = df[(df['time']>datetime(year=2019, month = 12, day=15, hour = 23))&(df['time']<datetime(year=2019, month = 12, day=23))]
    df52 = df[(df['time']>datetime(year=2019, month = 12, day=22, hour = 23))&(df['time']<datetime(year=2019, month = 12, day=30))]
    df1 = df[(df['time']>datetime(year=2019, month = 12, day=29, hour = 23))&(df['time']<datetime(year=2020, month = 1, day=6))]

    ds48 = ds[ds['time']<datetime(year=2019, month = 12, day=2)]
    ds49 = ds[(ds['time']>datetime(year=2019, month = 12, day=1, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=9))]
    ds50 = ds[(ds['time']>datetime(year=2019, month = 12, day=8, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=16))]
    ds51 = ds[(ds['time']>datetime(year=2019, month = 12, day=15, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=23))]
    ds52 = ds[(ds['time']>datetime(year=2019, month = 12, day=22, hour = 23))&(ds['time']<datetime(year=2019, month = 12, day=30))]
    ds1 = ds[(ds['time']>datetime(year=2019, month = 12, day=29, hour = 23))&(ds['time']<datetime(year=2020, month = 1, day=6))]

    background = []
    measurement = []
    prior = []
    conc = []
    for ds in [ds48, ds49, ds50, ds51, ds52, ds1]: 
        background.append(ds['background_inter'].mean())
        measurement.append(ds['xco2_measurement'].mean())
        prior.append(ds['xco2_inter'].mean())
    for df in [df48, df49, df50, df51, df52, df1]: 
        conc.append(df['conc'].mean())
    #print(background)
    #print(prior)


    weeks = [48, 49, 50, 51, 52, 53]
    plt.plot(weeks, background, marker = 'o', label = 'Background')
    plt.plot(weeks, measurement, marker = 'o', label = 'Measurement')
    plt.plot(weeks, prior,  marker = 'o',label = 'Prior')
    plt.plot(weeks, conc,  marker = 'o',label= 'Posterior')

    #ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    #ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    #df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    #ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
    plt.legend()
    #plt.xticks([datetime(year =2019, month = 12, day = 1),datetime(year =2019, month = 12, day = 5), datetime(year =2019, month = 12, day = 10), datetime(year =2019, month = 12, day = 15),
    #            datetime(year =2019, month = 12, day = 20), datetime(year =2019, month = 12, day = 25), datetime(year =2019, month = 12, day = 30), 
    #            datetime(year = 2020, month = 1, day = 4)], 
    #            rotation=45)
    plt.xticks([48, 49, 50, 51, 52, 53])
    ax.set_xticklabels(['48', '49', '50', '51', '52', '1'])
    plt.xlim((47.5, 53.5))
    #plt.xlim(left =  datetime(year = 2019, month = 11, day=30), right = datetime(year = 2020, month = 1, day=8))
    plt.grid(axis = 'both')
    plt.ylabel('concentration ['+unit+']')
    #plt.errorbar(x= datetime(year =2020, month = 1, day = 7), y = 407.5, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
    plt.xlabel('weeks')
    plt.title('Weekly mean concentration')
    myFmt = DateFormatter("%Y-%m-%d")
   # plt.xaxis.set_major_formatter(myFmt)
    ## Rotate date labels automatically
    #plt.autofmt_xdate()
    #ax.legend()
    #ax.grid(axis = 'both')
    #ax.set_ylabel('concentration [ppm]')
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_weekly_mean_concentrations_results.png')

    return

def calc_errors_per_region(flux_err, area_bioreg): 
    #flux_mean, flux_err = Inversion.get_flux()
    #ak_based2
    #area_bioreg = np.array([8.956336e+13, 5.521217e+10,2.585615e+11,9.965343e+10, 1.031680e+11, 6.341896e+10, 8.597746e+10, 
    #                      2.605820e+11,3.400408e+11, 6.450627e+12 ])
    
    #area_bioreg = np.array([8.956336e+13, 5.521217e+10,2.585615e+11,9.965343e+10, 1.031680e+11, 6.341896e+10, 1.075758e+11, 
    #                                                2.605820e+11,3.400408e+11, 6.450627e+12 ])

    flux_err_mean = np.zeros(len(flux_err.bioclass.values))
    for j in range(len(flux_err.bioclass.values)): 
        flux_err_mean[j] = flux_err[j,:].mean().values

    area_bioreg[0] = area_bioreg[0]*10000000#0000000 # make error for Ocean smaller
    res = flux_err_mean * (1/area_bioreg)
    res_scaled = np.zeros((len(flux_err.bioclass.values), len(flux_err.week.values)))
    for i in range(len(flux_err.week.values)):
        res_scaled[:,i] = res
    
    err_scaled = xr.DataArray(data=res_scaled, coords=dict({ 'bioclass': ('bioclass', [0,1,2,3,4,5,6]),
                                                                  'week': ('week', [1,48,49,50,51,52])}) )
    # set for every week the same error, because Christopher did it 

    return err_scaled

def calc_emission_factors(savepath,Inversion, predictions, datapathCO, datapathCO2, alpha, week_min, week_max): 
    means = []
    std = []
    #savepath = savepath+str(alpha)+'/'
    fire_pred_co = predictions.where(predictions.bioclass>0,drop=True)
    print(fire_pred_co)
    AU_mask = Inversion.map_on_grid(fire_pred_co[fire_pred_co['week']==50])
    print(AU_mask)
    AU_mask.values = np.ones(shape = AU_mask.shape)
    print(AU_mask)
    for wnum, week in enumerate([1,48,49,50,51,52]): 
        #ds_co = xr.open_dataset(datapathCO+'{:e}'.format(alpha)+'_CO_less_AK_1e-2_selected_results_week_'+str(week)+'.nc')
        #ds_co2 = xr.open_dataset(datapathCO2+'{:e}'.format(alpha)+'_CO2_less_AK_1e-2_selected_results_week_'+str(week)+'.nc')
        #ds_co = xr.open_dataset(datapathCO+'prior_spatially_week_'+str(week))
        #ds_co2 = xr.open_dataset(datapathCO2+'prior_spatially_week_'+str(week))

        ds_co, ds_co2, ds_co_dropped, ds_co2_dropped = select_region_ak_based(alpha, datapathCO, datapathCO2, week, wnum)
        #print(ds_co)
        #ds_co_dropped = xr.where(ds_co.__xarray_dataarray_variable__<=0.5, ds_co_dropped,False)
        #ds_co2_dropped = xr.where(ds_co.__xarray_dataarray_variable__<=0.5,  ds_co2_dropped,False)
        #ratio_dropped = ds_co_dropped.__xarray_dataarray_variable__ / ds_co2_dropped.__xarray_dataarray_variable__

        ds_co = ds_co.where(ds_co.__xarray_dataarray_variable__.values>0.5)
        ds_co2 = ds_co2.where(ds_co.__xarray_dataarray_variable__.values>0.5)
        #print(ds_co)


        ratio = ds_co.__xarray_dataarray_variable__ /ds_co2.__xarray_dataarray_variable__ *1e3#mikro g carbon/mirko g C
        #print(ratio)
        means.append(ratio.mean())
        std.append(ratio.std())
        plt.figure()
        plt.rcParams.update({'font.size':14})
        ax = plt.axes(projection=ccrs.PlateCarree())  
        my_cmap = LinearSegmentedColormap.from_list('', ['white', 'papayawhip'])
        AU_mask.plot(x='longitude', y='latitude', cmap = my_cmap)
        ratio.plot(x='longitude', y='latitude', cmap = 'seismic', vmax=20, vmin = -20, cbar_kwargs = {'label' : f'$\Delta$ CO / $\Delta$ CO$_2$ [ppb/ppm]', 'shrink': 0.8})
        plt.title('Fire emission ratios 2019 week '+str(week))
        ax.coastlines()
        plt.ylim((-45, -10))
        plt.xlim((110, 155))
        plt.savefig(savepath+'prior_co_co2_bbox_tight_week_'+str(week)+'.png', bbox_inches = 'tight')#AK_and_CO_5e-1_selected_co_co2_week_'+str(week)+'.png')
       # plt.savefig(savepath+'less_AK_1e-2_selected_fire_emission_ratios_co2_co_6e-1_week_'+str(week)+'.png')
    plt.figure()
    #plt.plot(np.arange(week_min, week_max+1,1), means)
    #plt.fill_between(np.arange(week_min, week_max+1,1), np.array(means)-np.array(std), np.array(means)+np.array(std), alpha = 0.5)
    plt.plot([1,48,49,50,51,52], means)
    plt.fill_between([1,48,49,50,51,52], np.array(means)-np.array(std), np.array(means)+np.array(std), alpha = 0.5)
    plt.ylim((-0.4, 0.4))
    plt.title('Weekly mean fire emission ratios')
    plt.savefig(savepath+'prior_co_co2.png')#'AK_and_CO_5e-1_selected_co2_co.png')
    #plt.savefig(savepath+'less_AK_1-2_selected_fire_ylim_0.4_emission_co2_co_ratios.png')

def select_region_bioclass_based(predictions_co):#, predictions_co2):
    fire_reg = list(np.arange(1, 700))
    '''fire_reg= [ 391,392,393,
                430,431,432,
                469,470,471,472,
                508,509,510,
                546, 547, 548, 549,
                583,584,585,586,
                613,614,615,
                637,638,
                656,657,658,
                667,668,669,670,
                677,678,679,680,
                684,685,686,687,688,689,690,691,692]'''
    #[312,313,314,315,
              #  350,351,352,353,354,
              #  389,390,391,392,393,
              #  428,429,430,431,432,
              #  466,467,468,469,470,471,472,
              #  505,506,507,508,509,510,
              ##  544,545, 546, 547, 548, 549,
               # 581,582,583,584,585,586,
               # 611,612,613,614,615,
               # 635,636,637,638,
               # 654,655,656,657,658,
               # 666,667,668,669,670,
               # 675,676,677,678,679,680,
               # 684,685,686,687,688,689,690,691,692]
    print(predictions_co)
    predictions_co.name = 'variable'
    fire_pred_co = predictions_co.where(predictions_co.bioclass==fire_reg[0], drop = True)
    for i in fire_reg: 
        pred_coi = predictions_co.where(predictions_co.bioclass==i, drop = True)
        fire_pred_co = xr.merge([pred_coi,fire_pred_co])
    #fire_pred_co2 = predictions_co2.where(predictions_co2.bioclass==fire_reg[0])
    #for i in fire_reg: 
    #    pred_coi = predictions_co2.where(predictions_co2.bioclass==i, drop = True)
    #    fire_pred_co2 = xr.merge([pred_coi,fire_pred_co2])
    print(fire_pred_co)
    
    return  fire_pred_co.variable

def select_region_ak_based(alpha, datapathCO, datapathCO2 , week, wnum): 
    #ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
    
    #for wnum, week in enumerate(week_list): 
    akco = xr.open_dataset(datapathCO+str("{:e}".format(alpha))+'_CO_ak_CO_spatially_week_'+str(week)+'.nc')
    akco2 = xr.open_dataset(datapathCO2+str("{:e}".format(alpha))+'_CO2_ak_CO2_spatially_week_'+str(week)+'.nc')
    
    ds_co = xr.open_dataset(datapathCO+'{:e}'.format(alpha)+'_CO_spatial_results_week_'+str(week)+'.nc')
    ds_co2 = xr.open_dataset(datapathCO2+'{:e}'.format(alpha)+'_CO2_spatial_results_week_'+str(week)+'.nc')
    ds_co_dropped = xr.where((akco<=1e-3)&(akco2<=1e-3),1, False)
    ds_co = ds_co.where((akco>1e-3)&(akco2>1e-3)).fillna(0)#&(akco2>1e-3))
    
    #print(ds_co)
    ds_co2 = ds_co2.where((akco>1e-3)&(akco2>1e-3))
    ds_co2_dropped = xr.where((akco<=1e-3)&(akco2<=1e-3), 1, False)


    return ds_co, ds_co2 , ds_co_dropped, ds_co2_dropped
        #ds_co.to_netcdf(datapathCO+'{:e}'.format(alpha)+'_CO_less_AK_1e-2_selected_results_week_'+str(week)+'.nc')
        #ds_co2.to_netcdf(datapathCO2+'{:e}'.format(alpha)+'_CO2_less_AK_1e-2_selected_results_week_'+str(week)+'.nc')
        #plt.savefig

def create_mask_ak_and_co_based(alpha, datapathCO, datapathCO2 , week, wnum):
    akco = xr.open_dataset(datapathCO+'test/'+str("{:e}".format(alpha))+'_CO_ak_CO_spatially_week_'+str(week)+'.nc')
    akco2 = xr.open_dataset(datapathCO2+str("{:e}".format(alpha))+'_CO2_ak_CO2_spatially_week_'+str(week)+'.nc')
    
    ds_co = xr.open_dataset(datapathCO+'test/'+str("{:e}".format(alpha))+'_CO_spatial_results_week_'+str(week)+'.nc')
    ds_co2 = xr.open_dataset(datapathCO2+'{:e}'.format(alpha)+'_CO2_spatial_results_week_'+str(week)+'.nc')
    ds_co_mask = xr.where((akco>1e-3)&(akco2>1e-3)&(ds_co.__xarray_dataarray_variable__.values>0.5),1, False)
    print(ds_co_mask)
    ds_co_mask.to_netcdf(datapathCO+'test/mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
    #ds_co = ds_co.where(ds_co.__xarray_dataarray_variable__.values>0.5)
    #ds_co = ds_co.where((akco>1e-3)&(akco2>1e-3)).fillna(0)#&(akco2>1e-3))
    
    #print(ds_co)
    #ds_co2_mask = ds_co2.where((akco>1e-3)&(akco2>1e-3))
    #ds_co2_dropped = xr.where((akco<=1e-3)&(akco2<=1e-3), 1, False)

        

def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num, 
                  err):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    l1, l2 = find_two_optimal_lambdas(Inversion,[1e-14,1], 1e-15)# 1e-8
    l3 = 1e-1
    for l in [l3]:#1, l2]:#,l2,l3]:
        predictions = Inversion.fit(alpha = l, xerr = err) 
        plot_l_curve(Inversion,err,molecule_name,savepath, l)
 
        print('Plotting spatial results')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion,molecule_name, week_min, week_max,l,vminv=None, diff =True)
        print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, molecule_name,week_min, week_max,l,vminv=None, diff =False)
        print('Plotting averaging kernels')
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=False)# class_num
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True)
        print('Plotting concentrations')
        calc_concentrations(Inversion,  'CO',l, savepath)
        plot_weekly_concentrations(Inversion,'CO',l, savepath)


savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Ecosystems/'#Setup_AK_based2/'
mask = "/home/b/b382105/ColumnFLEXPART/resources/bioclass_mask1.nc"
non_equal_region_size = True
area_bioreg = np.array([8.955129e+13, 5.576214e+11,4.387605e+12,1.708492e+12,4.420674e+11,2.976484e+11,3.358748e+11])

#for CO: 
Inversion = InversionBioclassCO(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl",#predictions_GFED_cut_tto_AU.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/",
    bioclass_path= mask, #'OekomaskAU_AKbased_2",#Flexpart_version8_all1x1
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)

flux_mean, flux_err = Inversion.get_flux()#Inversion.get_GFED_flux()
if non_equal_region_size == True: 
    print('per region')
    err = calc_errors_per_region(flux_err, area_bioreg)
    err = err*10**13*0.6 # scale such that mean**2 is equal to xprior_mean**2 - 100% Fehler 
    print('maximum error value: '+str(err.max()))
else: 
    print('not per region')
    err = Inversion.get_land_ocean_error(1/100000)
    print('maximum error value: '+str(err.max()))
print('Initlaizing done')
#print(err.mean())
predictions = Inversion.fit(alpha = 1e-1, xerr = err) 
#print(predictions.mean())
#print(Inversion.reg.y.mean())
#print(Inversion.reg.y.std())
print(Inversion.reg.x_prior.mean())
#print((Inversion.reg.x_prior.mean())**2)
#print(Inversion.reg.x_prior.std())
#print(Inversion.reg.x_covariance.mean())
#print(Inversion.reg.x_covariance.std())
#print(Inversion.reg.y_covariance.mean())
#print(Inversion.reg.y_covariance.std())
#print(Inversion.flux_errs.mean())
#print(Inversion.flux_errs_flat.mean())
print('predictions')
print(Inversion.predictions_flat.mean())
print(Inversion.predictions_flat)
print(err.values.flatten().mean())
xerr = err
xerr = xerr.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables]).values  

print('xerr')
print(xerr)
print(xerr.mean())
print('err')
print(err.values.flatten())
print('prior')
print(Inversion.reg.x_prior)

print('calculations')
print((((Inversion.reg.x_prior-Inversion.predictions_flat)/xerr)**2))
print((((Inversion.reg.x_prior-Inversion.predictions_flat)/xerr)**2).mean())
print((((Inversion.reg.x_prior-Inversion.predictions_flat)/xerr)**2).std())
print((((Inversion.reg.y - Inversion.reg.K@Inversion.reg.x_prior)/Inversion.concentration_errs)**2).mean())






########## do something else you might want to delete later #########
#class_num = plot_input_mask(savepath,mask)

########### do_everything ##################
#do_everything(savepath, Inversion, 'CO',mask, 1, 1, 6, err)
do_everything(savepath, Inversion,'CO',mask, 48, 52, 6, err)

########### emission factors ################
'''
predictions = Inversion.fit(alpha = 1e-5, xerr = err) 
calc_emission_factors('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/Ratios/',Inversion, predictions,
    '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/',
                      '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/',
                       1e-5, 1, 1)

calc_emission_factors('/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/Ratios/',Inversion, predictions, 
   '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/',
                      '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/',
                       1e-5, 48, 52)
'''
###### further not so relevant stuff #########

#for wnum, week in enumerate([48,49,50,51,52,1]):
#    create_mask_ak_and_co_based(1e-5, '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/', 
#                              '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/' , week, wnum)

#select_region_ak_based(1e-5, '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/',
#                        '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/' )


# for gridded : Optimal lambda 1: 1.271463009778368e-05
#Optimal lambda 2: 1.2802671525246307e-05
#Optimal lambda 3: 0.0006228666734243309
#Optimal lambda 4: 0.0009414995908741231

# for CO2: 
'''
Inversion = InversionBioclass(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path= "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_AKbased_2", #'OekomaskAU_AKbased_2",#Flexpart_version8_all1x1
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)
'''