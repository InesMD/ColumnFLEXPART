   
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
from matplotlib import colors as mcolors
from columnflexpart.scripts.plotting_routine_coupled_split_inversion import plot_spatial_result

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
    return hlines, vlines
    #colors = [mcolors.to_rgba(c)
    #      for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    #ax.add_collection(LineCollection(lines, **kwargs))


#####Plotting input mask ####
def plot_input_mask(savepath,datapath_and_name, selection= None): 
    plt.rcParams.update({'font.size':35})
    ds = xr.open_dataset(datapath_and_name)
    if selection != None: 
        ds0 = ds.where(ds.bioclass==selection[0])
        for i in selection: 
            dsi = ds.where(ds.bioclass==i, drop = True)
            ds0 = xr.merge([ds0,dsi])
        ds = ds0
    ds_ecosystem = xr.open_dataset('/home/b/b382105/ColumnFLEXPART/resources/bioclass_mask1.nc')
    plt.figure(figsize=(14, 10))    
    #plt.rcParams.update({'font.size':25})    
    ax = plt.axes(projection=ccrs.PlateCarree())   
    ds['bioclass'].plot(x = 'Long', y = 'Lat',cmap = 'nipy_spectral',add_colorbar = True, cbar_kwargs = {'shrink':0.865, 'label':'region number'})
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    ax.coastlines()
    plt.savefig(savepath+'bioclasses.png', bbox_inches = 'tight')

    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(1, 1, 1)
    #ax = plt.axes(projection=ccrs.PlateCarree())  
    mask = ds 
    #dsp = ds.where(ds.bioclass.values > 0)
    colors = ['lightblue', 'seagreen', 'chocolate','darkseagreen', 'mediumseagreen','forestgreen', 'darkgreen']
    cmap = LinearSegmentedColormap.from_list("", colors)
    #ds_ecosystem['bioclass'].plot(x = 'Long', y = 'Lat',add_colorbar = False, cmap = cmap)
    #hlines = np.array([[[]]])
    #vlines = np.array([[[]]])
    hlines, vlines = add_iso_line(ax = ax, x = mask.Long.values-0.5, y = mask.Lat.values-0.5, z = mask.bioclass.values, value = 0+0.5, color="black", linewidth=0.2)
    for b in list(set(list(ds.bioclass.values.flatten())))[1:-1]: # werden alle aufeinander geplotted - manche grau 
        hline, vline = add_iso_line(ax = ax, x = mask.Long.values-0.5, y = mask.Lat.values-0.5, z = mask.bioclass.values, value = b+0.5, color="black", linewidth=0.2)
        #print(hline[:])
        hlines = np.append(hlines, hline, axis = 0)
        vlines = np.append(vlines, vline, axis = 0)
    lines = np.vstack((vlines, hlines))
    #colors = np.ones(lines.shape[0])
    color = [mcolors.to_rgba("red") for x in range(lines.shape[0])]
    ax.add_collection(LineCollection(lines,  colors=color, linewidth=0.2, alpha = 1))
    ax.autoscale()
    #fig = ds['bioclass'].plot.contour(x = 'Long', y = 'Lat',cmap = 'nipy_spectral', row = ['x'], col = ['y'])#,ax = ax ,cmap = 'nipy_spectral', cbar_kwargs = dict(label='region number', shrink = 0.88))#gist_ncar
    #fig.cbar.set_label(label='region number', size = 20)
    ax.set_xlim((110.0, 155.0))
    ax.set_ylim((-45.0, -10.0))
    #ax.coastlines()
    if selection!=None: 
        plt.savefig(savepath+'less_selected_bioclasses.png')
    else:
        print('Region labels: '+str(set(ds.bioclass.values.flatten())))
        print('Number of regions: '+str(len(set(ds.bioclass.values.flatten()))))
        plt.savefig(savepath+'bioclasses_on_eco_test.png')

    return len(set(ds.bioclass.values.flatten()))
    


### Plotting spatial prior ########
def plot_prior_spatially(Inversion, molecule_name, week_min, week_max, savepath):
    plt.rcParams.update({'font.size':15})    
    factor = 10**6*12 
    flux_mean = Inversion.flux*factor

    for week in range(week_min, week_max+1):
        spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
        spatial_flux.to_netcdf(savepath+'prior_spatially_week_'+str(week))
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  
        spatial_flux.plot(x = 'longitude', y = 'latitude',  cmap = 'seismic', ax = ax, vmax = 0.2, vmin = -0.2, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.835})
        plt.scatter(x = 150.8793,y =-34.4061,color="black")
        ax.coastlines()
        #plt.title('Week '+str(week))
        plt.savefig(savepath+'Prior_'+molecule_name+'_mean_flux_spatial_week'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)

def plot_l_curve(Inversion,err, molecule_name, savepath, alpha):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]

    #1e-60,1e-55,1e-52,1e-50,1e-45,1e-40,1e-38, 1e-36, 1e-35,1e-34,1e-33,1e-32,1e-31, 5e-31, 2e-30,1e-29,5e-29, 1e-28,5e-28,1e-27,1e-26,5e-26, 1e-25,5e-25,1e-24, 1e-23,1e-22,1e-21, 1e-20,1e-19,1e-18, 1e-17,
    print('compute l curve')
    inv_result = Inversion.compute_l_curve(cond = 1e-14,alpha =[5e-9,1e-8,4e-8,1e-7,4e-7, 1e-6,4e-6,1e-5,4e-5, 1e-4,4e-4,1e-3,4e-3,1e-2,4e-2, 1e-1,4e-1,7e-1, 1,2,3,4,7, 10], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [3e-19,1e-18,3e-18,1e-17,3e-17,1e-16,3e-16,1e-15,3e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 1e-11,3e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,
    #                                                5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    #inv_result = Inversion.compute_l_curve(alpha = [1e-17,5e-17,1e-16,5e-16,1e-15,5e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 5e-12,1e-11,5e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    print('Plotting')
    plt.figure(figsize=(10,8))
    #print(inv_result)
    #plt.scatter(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    #plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    plt.scatter(inv_result["loss_forward_model"],inv_result["loss_regularization"], color = 'black')
    plt.scatter(inv_result["loss_forward_model"][10], inv_result["loss_regularization"][10], color = 'firebrick', label = '$\lambda = 4 *10^{-4}$')
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

def calculate_averaging_kernel(Inversion, class_num, week_num): 
    ak = Inversion.get_averaging_kernel()

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
    
    return ak_final, ak_sum

def multiply_prior_with_1_minus_averaging_kernel(Inversion, week_min, week_max, week_num, class_num, alpha, molecule_name, savepath): 
    # averaging kernel 
    ak_final, ak_sum = calculate_averaging_kernel(Inversion, class_num, week_num)
    ak_diag_sum = ak_sum.diagonal()[1:]
    ak_diag_final = ak_final.diagonal()[1:]
    #prior
    factor = 10**6*12 
    flux_mean = Inversion.flux*factor
    for week_idx, week in enumerate(range(week_min, week_max+1)):
        spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])

        #ak per week 
        ak_xr = xr.DataArray(data = ak_sum[week_idx*class_num:(week_idx+1)*class_num].diagonal()[1:], dims = ['bioclass'], 
                             coords=dict(bioclass= list(np.arange(1,class_num))))

        ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
        ones_matrix_with_shape_of_ak_xr = xr.DataArray(data = np.ones(ak_sum[week_idx*class_num:(week_idx+1)*class_num].diagonal()[1:].shape),
                                                        dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
        ones_matrix_shape_spatial_ak = Inversion.map_on_grid_without_time_coord(ones_matrix_with_shape_of_ak_xr, class_num)

        #multiplication
        prior_times_ak = spatial_flux*(ones_matrix_shape_spatial_ak-ak_spatial) 
        #plotting
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  

        prior_times_ak.plot( x='longitude', y='latitude', cmap = 'seismic', ax = ax, vmax = 0.2, vmin = -0.2,  cbar_kwargs = {'shrink':0.835})
        plt.scatter(x = 150.8793,y =-34.4061,color="black")
        ax.coastlines()
        plt.title('CO')
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'prior_times_1_minus_ak_week_'+str(week)+'.png', bbox_inches = 'tight')




def plot_averaging_kernel(Inversion, molecule_name, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    #result = Inversion.compute_l_curve(alpha = [0.004317500056754969], xerr = err)
    ak_final, ak_sum = calculate_averaging_kernel(Inversion, class_num, week_num)
    plt.rcParams.update({'font.size':15})
    if plot_spatially ==  True and weekly == False: 
        ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  
        orig_map=plt.cm.get_cmap('gist_heat')
        reversed_map = orig_map.reversed()
        ak_spatial.plot(norm = LogNorm(vmin = 1e-3, vmax = 6.8e-1), x='longitude', y='latitude', ax = ax,cmap = reversed_map,cbar_kwargs = {'shrink':0.835})#norm = LogNorm(vmin = 1e-3),
        plt.scatter(x = 150.8793,y =-34.4061,color="black")
        ax.coastlines()
        #plt.title('CO')
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_ak_spatial_gist_heat_reversed'+'log_1e-3_no_ocean.png', bbox_inches = 'tight')


    #sk_shape # Seite 47
    elif plot_spatially == True and weekly == True: 
        week_list = [1,48,49,50,51,52]
        for week in range(0,week_num): 
            ak_xr = xr.DataArray(data = ak_sum[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            #ak_xr = select_region_bioclass_based(ak_xr)
            ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
            ak_spatial.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_CO_spatially_week_'+str(week_list[week])+'.nc')
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())  
            orig_map=plt.cm.get_cmap('gist_heat')
            reversed_map = orig_map.reversed()
            ak_spatial.plot( x='longitude', y='latitude',ax = ax, vmax = 1,vmin = 0, cmap = reversed_map, cbar_kwargs = {'shrink':0.835, 'label': 'averaging kernel diagonal'})#norm = LogNorm(vmin = 1e-3),
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('CO')# week '+str(week_list[week])+' 2019 ')
            fig.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_no_log_gist_spatial_week'+str(week_list[week])+'_no_ocean.png', bbox_inches = 'tight', dpi = 450)
        #ak_xr = xr.DataArray(data = ak_sum[class_num:2*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[2*class_num:3*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[0:class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
    else: 
        plt.figure()
        plt.imshow(ak_final, vmin = 0, vmax = 1) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_ak_final_2d.png')
 

def find_two_optimal_lambdas(Inversion, range, stop, xerr):
    l = optimal_lambda(Inversion,range, stop, xerr)
    print('Optimal lambda 1: '+str(l))
    l2 = optimal_lambda(Inversion,[l, range[1]], stop, xerr)
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
            plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'Diff_to_prior_week_'+str(week)+'.png', bbox_inches = 'tight')
            plt.close()

        else: 
            spatial_result = (spatial_result*factor)
            #spatial_result = spatial_result.where(spatial_result.values>0.5)
            #print(week)
            #print(spatial_result.min())
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -10, vmax = 10, cbar_kwargs = {'label' : r'weekly flux [$\mu$gC m$^{-2}$ s$^{-1}$]', 'shrink':  0.835})
            plt.scatter(x = 150.8793,y =-34.4061,color="black")
            ax.coastlines()
            #plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_Spatial_results_week_'+str(week)+'.png', bbox_inches = 'tight', dpi = 450)
            spatial_result.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_spatial_results_week_'+str(week)+'.nc')
            plt.close()

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

    concentration_prior = Inversion.footprints_flat.values*Inversion.flux_flat.values*factor
    conc_sum_prior = concentration_prior.sum(axis = 1)
    conc_tot_prior = conc_sum_prior + ds['background_inter']


    plt.rcParams.update({'font.size':18})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (12,8))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])
    df.insert(loc = 1, column = 'prior', value = conc_tot_prior.values )
    df.insert(loc = 1, column = 'background_inter', value = ds['background_inter'] )
   
    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    df.plot(x = 'time', y = 'prior',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'salmon',label = 'Model prior')
    df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    
    #ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
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
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_concentrations_results_prior_without_ak.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

    return df

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
    plt.close()

    return




def calc_errors_per_region(flux_err, area_bioreg): 
    #flux_mean, flux_err = Inversion.get_flux()
    #ak_based2
    #area_bioreg = np.array([8.956336e+13, 5.521217e+10,2.585615e+11,9.965343e+10, 1.031680e+11, 6.341896e+10, 8.597746e+10, 
    #                      2.605820e+11,3.400408e+11, 6.450627e+12 ])
    
    #area_bioreg = np.array([8.956336e+13, 5.521217e+10,2.585615e+11,9.965343e+10, 1.031680e+11, 6.341896e+10, 1.075758e+11, 
    #                                                2.605820e+11,3.400408e+11, 6.450627e+12 ])

    ######################## with Christophers flux_err given in this function instead of flux_mean 
    flux_err_mean = np.zeros(len(flux_err.bioclass.values))
    for j in range(len(flux_err.bioclass.values)): 
        flux_err_mean[j] = flux_err[j,:].mean().values

    
    area_bioreg[0] = area_bioreg[0]*10000000#0000000 # make error for Ocean smaller
    res = flux_err_mean * (1/area_bioreg)
    res_scaled = np.zeros((len(flux_err.bioclass.values), len(flux_err.week.values)))
    for i in range(len(flux_err.week.values)):
        res_scaled[:,i] = res 
    err_scaled = xr.DataArray(data=res_scaled, coords=dict({ 'bioclass': ('bioclass',flux_err.bioclass.values),# [0,1,2,3,4,5,6]),
                                                                  'week': ('week',flux_err.week.values)}))# [1,48,49,50,51,52])}) )
    return err_scaled
    # set for every week the same error, because Christopher did it 
    #scale later with factor to match mean value 


    #################flux_mean, mean over bioclass: 
    #area_bioreg[0] = area_bioreg[0]*10000000
    #err_scaled = np.zeros((len(flux_mean.bioclass.values), len(flux_mean.week.values)))

    #for w in range(len(flux_mean.week.values)):
    #    mean_weekly_flux = flux_mean[1:,w].mean().values
    #    mean_spatial_weekly_flux = np.ones(len(flux_mean.bioclass.values))*mean_weekly_flux
    #    err_scaled[:,w] = mean_spatial_weekly_flux/area_bioreg

    #factor = flux_mean[1:].mean()/err_scaled[1:].mean()
    #err_scaled = err_scaled*factor.values



def calc_flux_mean_err_per_region(flux_mean, area_bioreg, mean_flux = None):
    ################## flux_mean area scaled: 

    area_bioreg[0] = area_bioreg[0]*10000000
    err_scaled = np.zeros((len(flux_mean.bioclass.values), len(flux_mean.week.values)))
    for w in range(len(flux_mean.week.values)):
        err_scaled[:,w] = flux_mean[:,w]/area_bioreg

    if mean_flux is None: 
        factor = flux_mean[1:].mean()/err_scaled[1:].mean()
        err_scaled = err_scaled*factor.values
    else: 
        factor = mean_flux[1:].mean()/err_scaled[1:].mean()
        err_scaled = err_scaled*factor.values
    #################################
    #area_array = 1/area_bioreg
    #M = np.dot(weekly_mean_flux_per_bioregion, area_array)
    err_scaled = xr.DataArray(data=err_scaled, coords=dict({ 'bioclass': ('bioclass',flux_mean.bioclass.values),# [0,1,2,3,4,5,6]),
                                                                  'week': ('week',flux_mean.week.values)}))
    return err_scaled


def calc_emission_factors(savepath,Inversion, predictions, datapathCO, datapathCO2, alpha, week_min, week_max): 
    means = []
    std = []
    #savepath = savepath+str(alpha)+'/'
    fire_pred_co = predictions.where(predictions.bioclass>0,drop=True)
    AU_mask = Inversion.map_on_grid(fire_pred_co[fire_pred_co['week']==50])
    AU_mask.values = np.ones(shape = AU_mask.shape)
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
    plt.close()

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
    predictions_co.name = 'variable'
    fire_pred_co = predictions_co.where(predictions_co.bioclass==fire_reg[0], drop = True)
    for i in fire_reg: 
        pred_coi = predictions_co.where(predictions_co.bioclass==i, drop = True)
        fire_pred_co = xr.merge([pred_coi,fire_pred_co])
    #fire_pred_co2 = predictions_co2.where(predictions_co2.bioclass==fire_reg[0])
    #for i in fire_reg: 
    #    pred_coi = predictions_co2.where(predictions_co2.bioclass==i, drop = True)
    #    fire_pred_co2 = xr.merge([pred_coi,fire_pred_co2])
    
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
    ds_co_mask.to_netcdf(datapathCO+'test/mask_co_co2_for_GFED_GFAS_week_'+str(week)+'.nc')
    #ds_co = ds_co.where(ds_co.__xarray_dataarray_variable__.values>0.5)
    #ds_co = ds_co.where((akco>1e-3)&(akco2>1e-3)).fillna(0)#&(akco2>1e-3))
    
    #print(ds_co)
    #ds_co2_mask = ds_co2.where((akco>1e-3)&(akco2>1e-3))
    #ds_co2_dropped = xr.where((akco<=1e-3)&(akco2<=1e-3), 1, False)

def plot_total_conc_with_errors(df, savepath, alpha, alphaCO): 
    # plotting 
    df = df.sort_values(['time'], ascending = True)
    df['date'] = pd.to_datetime(df['time'], format = '%Y-%M-%D').dt.date
    ds_mean = pd.read_pickle('/work/bb1170/RUN/b382105/Flexpart/TCCON/preparation/one_hour_runs/TCCON_mean_measurements_sept_to_march.pkl')
    #ds_mean = ds_mean[(ds_mean['datetime']>=datetime.datetime(year=2019, month =9, day=1))&(ds_mean['datetime']<= datetime.datetime(year=2020, month =2, day=28))]
    df= df[(df['time']>=datetime(year=2019, month =12, day=1))&(df['time']<= datetime(year=2020, month =1, day=1))].reset_index()

    ds_mean = ds_mean[(ds_mean['datetime']>=datetime(year=2019, month =12, day=1))&(ds_mean['datetime']<= datetime(year=2020, month =1, day=1))].dropna().sort_values(['datetime'], ascending = True).reset_index()
    print(ds_mean)
    df.insert(loc = 0, column = 'number_of_measurements', value = ds_mean['number_of_measurements'])

    plt.rcParams.update({'font.size':20})   
    fig, ax = plt.subplots(2,1,sharex = True, gridspec_kw={'height_ratios': [4,0.8]},figsize=(18,6))
    Xaxis = np.arange(0,len(df['time'][:]),1)

    ax[0].set_xticks(Xaxis, ['']*(len(df['time'][:])))
    ax[0].set_xlim((-1,len(df['time'][:])))

    ## CO plot
    ax2 = ax[0]#.twinx()
    ax2.set_ylabel(r'CO [ppb]')
    max_value = max(abs(df['conc'].max()), abs(df['conc'].min()))
    ax2.set_ylim((5, 335))
    # total CO
    ax2.plot(Xaxis, df['xco2_measurement'],color = 'dimgrey',label = r'measurements')
    ax2.plot(Xaxis, df['background_inter'], color = 'k', label = 'background')
    lns1 = ax2.plot(Xaxis,df['prior'],color = 'salmon',  label = r'total prior')
    ax2.fill_between(Xaxis, df['prior']-(df['prior_std']), df['prior']+(df['prior_std']), color = 'salmon', alpha = 0.3)
    lns1 = ax2.plot(Xaxis,df['conc'],color = 'red',  label = r'total posterior')
    ax2.fill_between(Xaxis, df['conc']-(df['post_std']), df['conc']+(df['post_std']), color = 'red', alpha = 0.5)
    ax2.legend(loc = 'upper left')
    # prior CO 
    ax[1].bar(Xaxis, df['number_of_measurements'], width=0.5, color = 'dimgrey')
    ax[1].set_ylabel('N', labelpad=17)
    ax[1].grid(axis = 'x')
    ax2.grid(axis = 'both')

    reference = df['date'][0].day
    ticklabel_list = [df['date'][0]]
    index_of_ticklabel_list = [0]
    for i in np.arange(1,len(df['time'][:])):
        if df['date'][i].day>reference:# and reference<12:
            index_of_ticklabel_list.append(i)
            if df['date'][i].day == 5: 
                ticklabel_list.append('')
            elif df['date'][i].day == 16: 
                ticklabel_list.append('')
            elif df['date'][i].day == 25: 
                ticklabel_list.append('')
            else: ticklabel_list.append(df['date'][i])
            #ticklabel_list.append(df['date'][i])
            reference = df['date'][i].day

    ax[1].set_xticks(index_of_ticklabel_list, ticklabel_list)
    ax[1].xaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabel_list))
    #ax[1].set_xlabel('day')
    plt.gcf().autofmt_xdate(rotation = 45, ha = 'right', bottom = 0) 
    plt.axhline(y=0, color='k', linestyle='-')
    plt.subplots_adjust(hspace=0)

    fig.savefig(savepath+"{:.2e}".format(alphaCO)+"_CO_total_concentrations_with_errors_measurement_entire_time_series.png", dpi = 300, bbox_inches = 'tight')


def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num, 
                  err):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[1e-7,10], 1e-14, err)# 1e-8

    l3 = 1e-1
    for l in [3e-2]:#1, l2]:#,l2,l3]:
        predictions = Inversion.fit(alpha = l, xerr = err)
        #plot_l_curve(Inversion,err,molecule_name,savepath, l)
        # plot errors: 
        for w in range(week_min, week_max+1): 
            plt.rcParams.update({'font.size': 14})
            plot_spatial_result(Inversion.map_on_grid(Inversion.prediction_errs[Inversion.prediction_errs['week']==w, :]*10**6*12), savepath, str("{:.2e}".format(l))+'_posterior_std_CO'+str(w)+'.png', 'pink_r', vmax =0.32, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation [$\mu$gC m$^{-2}$s$^{-1}$]'}, norm = None )
            plot_spatial_result(Inversion.map_on_grid(err[:,err['week'] == w]/np.sqrt(l)*10**6*12), savepath, str("{:.2e}".format(l))+'_prior_std_CO'+str(w)+'.png', 'pink_r', vmax =0.32, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'standard deviation [$\mu$gC m$^{-2}$s$^{-1}$]'}, norm = None )
            plot_spatial_result(Inversion.map_on_grid(((err[:,err['week'] == w]/np.sqrt(l)*10**6*12)-(Inversion.prediction_errs[Inversion.prediction_errs['week']==w, :]*10**6*12))/(err[:,err['week'] == w]/np.sqrt(l)*10**6*12)), savepath, str("{:.2e}".format(l))+'_uncertainty_red_CO_'+str(w)+'.png', 'bone_r', vmax =1, vmin =0, 
                                    cbar_kwargs = {'shrink':  0.835, 'label' : r'uncertainty reduction'}, norm = None )
            post_err = Inversion.prediction_errs[Inversion.prediction_errs['week']==w, :]*10**6*12
            Inversion.map_on_grid(Inversion.prediction_errs[Inversion.prediction_errs['week']==w,:]*10**6*12).to_netcdf(savepath+str(l)+'_CO_spatial_posterior_error_week'+str(w)+'.nc')
            post_err.to_netcdf(savepath+str(l)+'_CO_posterior_error_week'+str(w)+'.nc')
            prior_err = err[:,err['week'] == w]/np.sqrt(l)*10**6*12
            prior_err.to_netcdf(savepath+str(l)+'_CO_prior_error_week_'+str(w)+'.nc')
            Inversion.map_on_grid(err[:,err['week'] == w]/np.sqrt(l)*10**6*12).to_netcdf(savepath+str(l)+'_CO_spatial_prior_error_week'+str(w)+'.nc')
            Inversion.predictions[Inversion.predictions['week']==w].to_netcdf(savepath+'results_bioclass_week'+str(w)+'.nc')

        print('Plotting spatial results')
        #plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion,molecule_name, week_min, week_max,l,vminv=None, diff =True)
        print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, molecule_name,week_min, week_max,l,vminv=None, diff =False)

        print('Plotting averaging kernels')
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=False)# class_num
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True, weekly=False)
        print('Plotting concentrations')
        df = calc_concentrations(Inversion,  'CO',l, savepath)
        #plot_weekly_concentrations(Inversion,'CO',l, savepath)
        posterior_conc = Inversion.prediction_errs_flat.values*Inversion.footprints_flat.values*10**9
        posterior_conc = posterior_conc.sum(axis = 1)
        df.insert(loc = 0, column = 'post_std', value=posterior_conc)
        prior_errs_flat = xr.DataArray(
            data = err.stack(new=[Inversion.time_coord, *Inversion.spatial_valriables]).values,
            dims = ["new"],
            coords = dict(new=Inversion.coords)
        )
        prior_errs_flat = prior_errs_flat.values*Inversion.footprints_flat.values*10**9/np.sqrt(l)
        prior_errs_flat = prior_errs_flat.sum(axis = 1)
        df.insert(loc = 0, column = 'prior_std', value=prior_errs_flat)
        plot_total_conc_with_errors(df, savepath, l, l)


def calc_errors_flat_area_weighted_scaled_to_mean_flux(flux_mean, area_bioreg): 
    flat_errors = np.ones((len(flux_mean.bioclass.values), len(flux_mean.week.values)))
    area_bioreg[0] = area_bioreg[0]*10000000 # ocean smaller
    final_error = np.ones(flat_errors.shape)
    for w in range(len(flux_mean.week.values)): 
        area_weighted_errors = flat_errors[:,w]/area_bioreg
    
        scaling_factor = flux_mean[1:, w].mean()/area_weighted_errors[1:].mean()
        #print(flux_mean[1:,w].mean())
        #print(scaling_factor.values)
        final_error[:,w] = scaling_factor.values*area_weighted_errors
    #print(final_error)
    err_scaled = xr.DataArray(data=final_error, coords=dict({ 'bioclass': ('bioclass',flux_mean.bioclass.values),# [0,1,2,3,4,5,6]),
                                                                'week': ('week',flux_mean.week.values)}))
    #print(err_scaled)
    return err_scaled
    


savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Ecosystems_AK_split_with_4/'#Setup_gridded/'#Setup_AK_based2/'cosystems_AK_split_with_4/'#
mask = "/home/b/b382105/ColumnFLEXPART/resources/Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#OekomaskAU_Flexpart_version8_all1x1"#Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#OekomaskAU_Flexpart_version8_all1x1"#"#bioclass_mask1.nc"Ecosystems_AK_based_split_with_21_and_20_larger_and_4_split.nc"#
non_equal_region_size = True



#### area_bioreg
#Ecosystems
area_bioreg = np.array([8.955129e+13, 5.576214e+11,4.387605e+12,1.708492e+12,4.420674e+11,2.976484e+11,3.358748e+11])
#Ecosystems, AK split
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.95434996e+11, 6.07622615e+10, 6.98377101e+10, 1.77802351e+11,
 1.95969571e+11])
#Ecosystems, AK split, with 21
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 9.89272615e+10, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
 1.77802351e+11, 1.95969571e+11])
#Ecosystemes, AK_split with 21 and 20 larger 
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.51305732e+12,
 4.42067398e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
 5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
 3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
 4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
 1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
 1.77802351e+11, 1.95969571e+11])
#area bioreg for reion 4 split 21 and 20 larger: 
area_bioreg = np.array([8.95512935e+13, 4.27021383e+11, 4.01383356e+12, 1.45736998e+12,
2.15745316e+11, 9.81395187e+09, 3.95111921e+10, 3.18183296e+10,
5.52121748e+10, 5.55915961e+10, 5.41460509e+10, 4.29887311e+10,
3.13764949e+10, 1.05335427e+10, 9.31980461e+10, 5.11758810e+10,
4.00158388e+10, 2.99495636e+10, 3.91250878e+10, 4.90667320e+10,
1.54614599e+11, 9.65077342e+10, 6.07622615e+10, 6.98377101e+10,
1.77802351e+11, 1.95969571e+11, 2.26322082e+11])

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
#err = calc_errors_flat_area_weighted_scaled_to_mean_flux(flux_mean, area_bioreg)
#error = np.ones((len(flux_mean.bioclass.values), len(flux_mean.week.values)))*flux_mean[1:,:].mean().values
#error[0,:] = error[0,:]*10**-7
#err = xr.DataArray(data=error, coords=dict({ 'bioclass': ('bioclass',flux_mean.bioclass.values),# [0,1,2,3,4,5,6]),
#                                                                'week': ('week',flux_mean.week.values)}))
 




if non_equal_region_size == True: 
    print('per region')
    err = calc_errors_flat_area_weighted_scaled_to_mean_flux(flux_mean, area_bioreg)
    #err_err_mean = calc_errors_per_region(flux_err, area_bioreg)
    #err_mean_mean = calc_errors_per_region(flux_mean, area_bioreg)
    #err_mean = calc_flux_mean_err_per_region(flux_mean, area_bioreg)
    #err_err = calc_flux_mean_err_per_region(flux_err, area_bioreg, flux_mean)
    #err_err_mean = err_err_mean*10**13*0.6   #*10**12*0.4      #*10**13*0.297#-50% err, *10**13*0.893#-150% err ; *10**13*0.6 - 100% err # scale such that xerr_mean**2 is equal to xprior_mean**2 
    #err_mean_mean = err_mean_mean*10**12*0.6
    #print(err_err_mean.max())
    print('maximum error value: '+str(err.max()))
else: 
    print('not per region')
    err = Inversion.get_land_ocean_error(1/100000)
    print('maximum error value: '+str(err.max()))
print('Initlaizing done')
'''
print(err)
plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())  
err = Inversion.map_on_grid(err[:,err['week']==52])[:]*10**6*12
err.plot(x = 'longitude', y = 'latitude', cmap = 'Oranges', ax = ax,  cbar_kwargs = {'label' : r'weekly flux error [$\mu$gC m$^{-2}$s$^{-1}$]', 'shrink': 0.935})
ax.coastlines()
plt.savefig(savepath+'prior_errors.png', dpi = 300)
'''
#print(err)
#print(err.mean())

#print(flux_err)

#plt.fill_between(flux_mean['bioclass'].values,(flux_mean[:,0].values+flux_err[:,0].values)*10**6*12 ,(flux_mean[:,0].values-flux_err[:,0].values)*10**6*12 , label = 'std', alpha = 0.5 )
#plt.plot(flux_err['bioclass'].values,flux_err[:,0].values*10**6*12 , label = 'std' )
#plt.plot(flux_err['bioclass'].values,flux_mean[:,0].values*10**6*12 , label = 'mean' )
#plt.plot(flux_err['bioclass'].values,err_err[:,0].values*10**6*12 , label = 'err: std area' )
#plt.plot(flux_err['bioclass'].values,err[:,0].values*10**6*12 , label = 'flat err' )
#plt.plot(err_mean['bioclass'].values, err_mean[:,0].values*10**6*12, label = 'err: mean area')
#plt.plot(flux_err['bioclass'].values,err_err_mean[:,0].values*10**6*12 , label = 'err: std mean area' )
##plt.plot(err_mean['bioclass'].values, err_mean_mean[:,0].values*10**6*12, label = 'err: mean mean area')

#plt.xlabel('bioclasses')
#plt.ylabel('prior flux mean and std [$\mu$gC m$^{-2}$s$^{-1}$]')
#plt.legend()
#plt.savefig(savepath+'flux_mean_and_flux_err_really_flat_and_not.png')
#print(flux_mean)


#predictions = Inversion.fit(alpha = 1e-1, xerr = err) 
#class_num = plot_input_mask(savepath,mask)
'''
multiply_prior_with_1_minus_averaging_kernel(Inversion, 48, 52, 6, class_num, 1e-1, 'CO', savepath)

#print(predictions.mean())
#print(Inversion.reg.y.mean())
#print(Inversion.reg.y.std())

#print(flux_mean.mean()**2)
print((Inversion.reg.x_prior.mean())**2)
#print((Inversion.reg.x_prior.mean())**2)
#print(Inversion.reg.x_prior.std())
print(Inversion.reg.x_covariance.mean())
#print(Inversion.reg.x_covariance.std())
#print(Inversion.reg.y_covariance.mean())
#print(Inversion.reg.y_covariance.std())
#print(Inversion.flux_errs.mean())
#print(Inversion.flux_errs_flat.mean())
print('predictions')
'''
'''
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

'''




########## do something else you might want to delete later #########
#class_num = plot_input_mask(savepath,mask)

########### do_everything ##################
do_everything(savepath, Inversion,'CO',mask, 48, 52, 6, err)
do_everything(savepath, Inversion, 'CO',mask, 1, 1, 6, err)


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