   
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
def plot_prior_spatially(Inversion, molecule_name, week_min, week_max, savepath):
    plt.rcParams.update({'font.size':13})    
    factor = 10**6*12 
    flux_mean = Inversion.flux*factor

    for week in range(week_min, week_max+1):
        spatial_flux = Inversion.map_on_grid(flux_mean[:,flux_mean['week']==week])
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  
        spatial_flux.plot(x = 'longitude', y = 'latitude',  cmap = 'seismic', ax = ax, vmax = 0.5, vmin = -0.5, cbar_kwargs = {'label' : r'flux [$\mu$ gCm$^{-2}$s$^{-1}$]'})
        ax.coastlines()
        plt.title('Week '+str(week))
        plt.savefig(savepath+'Prior_'+molecule_name+'_mean_flux_spatial_week'+str(week)+'.png')

def plot_l_curve(Inversion,err, molecule_name, savepath, alpha):
    #[1e-8,3e-8, 1e-7, 4.9e-7, 1e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 5e-4, 1e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1]
    # [1e-8,2.5e-8,5e-8,1e-7,2.5e-7,5.92e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1]
    print('compute l curve')
    #inv_result = Inversion.compute_l_curve(alpha =[1e-60,1e-50,1e-45,1e-40,1e-38, 1e-35,1e-33, 1e-30, 1e-28,2e-27, 1e-25, 1e-23, 1e-20, 1e-17, 1e-16,1e-15, 1e-14, 1e-13, 1e-12,1e-11,3e-11,1e-10,4e-10,1e-9,2.16e-9,4e-9,1e-8,3e-8, 1e-7, 2e-7, 3e-7, 7.05e-7, 1.11e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2.6e-4, 3e-4, 6.23e-4, 8.92e-4,1.87e-3,3.3e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1], xerr = err)
    inv_result = Inversion.compute_l_curve(alpha = [3e-19,1e-18,3e-18,1e-17,3e-17,1e-16,3e-16,1e-15,3e-15,1e-14, 5e-14,1e-13, 5e-13, 1e-12, 1e-11,3e-11,1e-10,5e-10,1e-9,4e-9,1e-8,2.5e-8,5e-8,1e-7,2e-7,5e-7,1e-6,2.5e-6,5e-6,1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,2.5e-3,4.32e-3,1e-2,2.5e-2,5e-2,1e-1,1.5e-1,5e-1,1], xerr = err)
    print('Plotting')
    plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    fig, ax = Inversion.plot_l_curve(mark_ind = 18, mark_kwargs= dict(color = 'firebrick',label = r'$\lambda =1 \cdot 10^{-9$'))
    plt.grid(axis = 'y', color = 'grey', linestyle = '--' )
    plt.legend()
    print('saving')
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_l_curve_xerr_extended2.png')


def plot_averaging_kernel(Inversion, molecule_name, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    #result = Inversion.compute_l_curve(alpha = [0.004317500056754969], xerr = err)
    ak = Inversion.get_averaging_kernel()
    plt.rcParams.update({'font.size':13})
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
        ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax,cmap = reversed_map)
        ax.coastlines()
        plt.title('Averaging kernel for 2019/12')
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_ak_spatial_gist_heat_reversed'+'log_1e-3_no_ocean.png')

    #sk_shape # Seite 47
    elif plot_spatially == True and weekly == True: 
        week_list = [1,48,49,50,51,52]
        for week in range(0,week_num): 
            ak_xr = xr.DataArray(data = ak_sum[week*class_num:(week+1)*class_num].diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))
            ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())  
            orig_map=plt.cm.get_cmap('gist_heat')
            reversed_map = orig_map.reversed()
            ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax, vmax = 1, cmap = reversed_map)
            ax.coastlines()
            plt.title('Averaging kernel for 2019 week '+str(week_list[week]))
            fig.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_ak_spatial_gist_heat_reversed_week'+str(week_list[week])+'_no_ocean.png')
        #ak_xr = xr.DataArray(data = ak_sum[class_num:2*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[2*class_num:3*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[0:class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
    else: 
        plt.figure()
        plt.imshow(ak_final, vmin = 0, vmax = 1) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_ak_final_2d.png')
 

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
    plt.rcParams.update({'font.size':13})   
    for week in range(week_min,week_max+1): 
        plt.figure()
        #for bioclass in [0,1,2,3,4,5,6,7,8,10]: 
        #       #print(predictions)
        #       pred = predictions[predictions['week']==week]
        #       #print(predictions)
        #       spat_result_class = Inversion.map_on_grid(pred[:,pred['bioclass']==bioclass])
        #       spat_result_class.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_spatial_results_week_'+str(week)+'_bioclass_'+str(bioclass)+'.nc')
        spatial_result = Inversion.map_on_grid(Inversion.predictions[Inversion.predictions['week']==week])
        ax = plt.axes(projection=ccrs.PlateCarree())  
        if diff== True: 
            spatial_flux = Inversion.map_on_grid(Inversion.flux[:,Inversion.flux['week']==week])
            spatial_flux = spatial_flux *factor
            spatial_result = (spatial_result*factor)-spatial_flux
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -50, vmax = 50, cbar_kwargs = {'label' : r'flux [$\mu$gC m$^{-2}$ s$^{-1}$]'})
            ax.coastlines()
            plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_err_per_reg_Diff_to_prior_week_'+str(week)+'_xerr.png')
        else: 
            spatial_result = (spatial_result*factor)
            spatial_result.plot(x = 'longitude', y = 'latitude',ax = ax, cmap = 'seismic',vmin = -50, vmax = 50, cbar_kwargs = {'label' : r'flux [$\mu$gC m$^{-2}$ s$^{-1}$]'})
            ax.coastlines()
            plt.title('Week '+str(week))
            plt.savefig(savepath+str("{:e}".format(alpha))+'_err_per_reg_Spatial_results_week_'+str(week)+'_xerr.png')
            spatial_result.to_netcdf(path =savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_spatial_results_week_'+str(week)+'.nc')
    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')



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

    plt.rcParams.update({'font.size':14})   
    plt.rcParams.update({'errorbar.capsize': 5})
    fig, (ax, ax2) = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [4, 1]},figsize = (14,10))
    df = pd.DataFrame(data =conc_tot.values, columns = ['conc'])
    df.insert(loc = 1, column = 'time', value = ds['time'])
    df.insert(loc = 1, column = 'measurement_uncertainty', value = ds['measurement_uncertainty'])
    df.insert(loc = 1, column = 'xco2_measurement', value = ds['xco2_measurement'])
   
    ds.plot(x='time', y='background_inter',color = 'k', marker='.', markersize = 7,linestyle='None', ax=ax, label= 'Model background')
    ds.plot(x='time', y='xco2_measurement',marker='.',ax=ax, markersize = 7,linestyle='None',color = 'dimgrey',label= 'Measurement')#yerr = 'measurement_uncertainty', 
    #plt.fill_between(df['time'],df['xco2_measurement']-df['measurement_uncertainty'],
    #                 df['xco2_measurement']+df['measurement_uncertainty'],color = 'lightgrey' )
    df.plot(x = 'time', y = 'conc',marker = '.', ax = ax, markersize = 7,linestyle='None',color = 'red',label = 'Model posterior')
    ds.plot(x='time', y='xco2_inter', marker='.', ax = ax,markersize = 7,linestyle='None',color = 'salmon', label='Model prior')
    ax.legend()
    ax.set_xticks([datetime(year =2019, month = 12, day = 1),datetime(year =2019, month = 12, day = 5), datetime(year =2019, month = 12, day = 10), datetime(year =2019, month = 12, day = 15),
                datetime(year =2019, month = 12, day = 20), datetime(year =2019, month = 12, day = 25), datetime(year =2019, month = 12, day = 30), 
                datetime(year = 2020, month = 1, day = 4)], 
                rotation=45)
    ax.set_xlim(left =  datetime(year = 2019, month = 11, day=30), right = datetime(year = 2020, month = 1, day=8))
    ax.grid(axis = 'both')
    ax.set_ylabel('concentration [ppm]')
    ax.errorbar(x= datetime(year =2020, month = 1, day = 7), y = 300, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
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
    ax2.set_ylabel('# measurements')
    ax2.grid(axis='x')
    plt.subplots_adjust(hspace=0)
    plt.savefig(savepath+str("{:e}".format(alpha))+'_'+molecule_name+'_concentrations_results.png')

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

def calc_errors_per_region(flux_err): 
    #flux_mean, flux_err = Inversion.get_flux()
    #ak_based2
    area_bioreg = np.array([8.956336e+13, 5.521217e+10,2.585615e+11,9.965343e+10, 1.031680e+11, 6.341896e+10, 8.597746e+10, 
                          2.605820e+11,3.400408e+11, 6.450627e+12 ])
    
    flux_err_mean = np.zeros(len(flux_err.bioclass.values))
    for j in range(len(flux_err.bioclass.values)): 
        flux_err_mean[j] = flux_err[j,:].mean().values

    area_bioreg[0] = area_bioreg[0]*1000000000000 # make error for Ocean smaller
    res = flux_err_mean * (1/area_bioreg)
    res_scaled = np.zeros((len(flux_err.bioclass.values), len(flux_err.week.values)))
    for i in range(len(flux_err.week.values)):
        res_scaled[:,i] = res
    
    err_scaled = xr.DataArray(data=res_scaled, coords=dict({ 'bioclass': ('bioclass', [0,1,2,3,4,5,6,7,8,10]),
                                                                  'week': ('week', [1,48,49,50,51,52])}) )
    # set for every week the same error, because Christopher did it 

    return err_scaled

def do_everything(savepath, Inversion, molecule_name, mask_datapath_and_name, week_min, week_max, week_num, 
                  err):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    plot_prior_spatially(Inversion,molecule_name,week_min, week_max, savepath)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[1e-10,1], 1e-11)# 1e-8
    #l1 =  0.0032959245592851234
    #l3 = 0.0018658549884632656
    #l2 =0.000892262853255476
    #l4 = 0.00026809900305699557
    #l1 = 4e-9
    #l2 =6.228667e-04
    #l1 = 5.1e-27
    #l2 = 5.2e-27
    #l2 = 1e-29
    l1 = 1e-9
    #l3= 1e-35
    #l2 = 1e-5
    #l3 = 1e-15
    for l in [l1]:#,l2]:#,l3]:
        #plot_l_curve(Inversion,molecule_name,savepath, l)
        predictions = Inversion.fit(alpha = l, xerr = err) 
        print('Plotting spatial results')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion,molecule_name, week_min, week_max,l,vminv=None, diff =True)
        print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, molecule_name,week_min, week_max,l,vminv=None, diff =True)
        print('Plotting averaging kernels')
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=False)# class_num
        plot_averaging_kernel(Inversion,molecule_name, l, class_num, week_num,savepath, plot_spatially=True,weekly = True)
        print('Plotting concentrations')
        calc_concentrations(Inversion,  'CO',l, savepath)
        plot_weekly_concentrations(Inversion,'CO',l, savepath)

savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images_CO/Setup_gridded/'#Setup_AK_based2/'
non_equal_region_size = False
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
#for CO: 
Inversion = InversionBioclassCO(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions3_CO.pkl",#predictions_GFED_cut_tto_AU.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CAMS/Fluxes/Regridded_1x1/",
    bioclass_path= "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1", #'OekomaskAU_AKbased_2",#Flexpart_version8_all1x1
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)

flux_mean, flux_err = Inversion.get_flux()#Inversion.get_GFED_flux()
if non_equal_region_size == True: 
    err = calc_errors_per_region(flux_err)
else: 
    err = Inversion.get_land_ocean_error(1/100000)
#print(err)
print('Initlaizing done')
#plot_prior_spatially(Inversion,'CO',48, 52, savepath)
#plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion,'CO', 48, 52,1e-29,vminv=None, diff =False)

#calc_errors_per_region(Inversion)



#l1,l2 = find_two_optimal_lambdas(Inversion,[1e-11,1], 1e-12)
#plot_l_curve(Inversion, savepath, 4e-9)
#print(l1)
#calc_concentrations(Inversion,   0.00026809900305699557,savepath)
#plot_averaging_kernel(Inversion,  0.0032959245592851234, 10, 6,savepath, True, True)
#calc_concentrations(Inversion,  2.616672e-04, savepath)
#plot_weekly_concentrations(Inversion, 'CO',1e-29, savepath)
#calc_concentrations(Inversion,  'CO',1e-5, savepath)
#plot_weekly_concentrations(Inversion,'CO',1e-5, savepath)


#plot_input_mask(savepath,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1", selection= regions_sensitive)
do_everything(savepath, Inversion, 'CO',"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
             1,1,6, err)
do_everything(savepath, Inversion,'CO',"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
              48,52,6, err)
plot_l_curve(Inversion,err,'CO', savepath,1e-9)

#plot_weekly_concentrations(Inversion, 5.2e-27, savepath)
#plot_weekly_concentrations(Inversion, e-26, savepath)
#calc_concentrations(Inversion, 4e-9, savepath)
#plot_weekly_concentrations(Inversion, 6.228667e-04, savepath)
#class_num = plot_input_mask(savepath,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania")#,selection = [5,6,7,8,16,20,31])
# for gridded : Optimal lambda 1: 1.271463009778368e-05
#Optimal lambda 2: 1.2802671525246307e-05
#Optimal lambda 3: 0.0006228666734243309
#Optimal lambda 4: 0.0009414995908741231