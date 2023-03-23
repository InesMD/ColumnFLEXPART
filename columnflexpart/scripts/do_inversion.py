   
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
from matplotlib.colors import LogNorm 
from matplotlib.dates import DateFormatter



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
    print('compute l curve')
    inv_result = Inversion.compute_l_curve(alpha =[1e-13, 1e-12,1e-11,1e-10,1e-9,1e-8,3e-8, 1e-7, 4.92e-7, 1.11e-6,2e-6,5e-6, 1e-5,2e-5, 3e-5,5e-5,1e-4, 2e-4, 3e-4, 6.2e-4, 1.41e-3,2e-3,4.32e-3, 1e-2,1.8e-2,3e-2,5e-2,7e-2, 1e-1,1.8e-1,3e-1,5e-1, 1])
    print('Plotting')
    plt.plot(inv_result["loss_regularization"],inv_result["loss_forward_model"])
    fig, ax = Inversion.plot_l_curve(mark_ind = 19, mark_kwargs= dict(color = 'firebrick',label = r'$\lambda = 6.2 \cdot 10^{-4}$'))
    plt.grid(axis = 'y', color = 'grey', linestyle = '--' )
    plt.legend()
    print('saving')
    plt.savefig(savepath+'l_curve_'+str(alpha)+'_extended.png')


#plot_l_curve(Inversion)
def plot_averaging_kernel(Inversion, alpha, class_num, week_num,savepath, plot_spatially, weekly = False):
    flux_mean, flux_err = Inversion.get_flux()
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) 
    print(predictions)
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
        print(ak_final.diagonal())
        print(Inversion.time_coord)
        ak_xr = xr.DataArray(data = ak_final.diagonal()[1:], dims = ['bioclass'], coords=dict(bioclass= list(np.arange(1,class_num))))#, week = [1,48,49,50,51,52])))
        print(ak_xr)
        ak_spatial = Inversion.map_on_grid_without_time_coord(ak_xr, class_num)
        print(ak_spatial)
        ax = plt.axes(projection=ccrs.PlateCarree())  
        orig_map=plt.cm.get_cmap('gist_heat')
        reversed_map = orig_map.reversed()
        ak_spatial.plot(norm = LogNorm(vmin = 1e-3), x='longitude', y='latitude',ax = ax,cmap = reversed_map)
        ax.coastlines()
        plt.title('Averaging kernel for 2019/12')
        plt.savefig(savepath+'ak_spatial_gist_heat_reversed'+str(alpha)+'log_1e-3_no_ocean.png')
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
            ak_spatial.plot(x='longitude', y='latitude',ax = ax, vmax = 0.2, vmin = 0, cmap = reversed_map)
            ax.coastlines()
            plt.title('Averaging kernel for 2019 week '+str(week_list[week]))
            fig.savefig(savepath+'ak_spatial_gist_heat_reversed'+str(alpha)+'_week'+str(week_list[week])+'_no_ocean.png')
        #ak_xr = xr.DataArray(data = ak_sum[class_num:2*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[2*class_num:3*class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))
        #ak_xr = xr.DataArray(data = ak_sum[0:class_num].diagonal()[1:], dims = ['bioclass', 'week'], coords=dict(bioclass= list(np.arange(1,class_num)), week = 1))

    #sensitive = np.where(ak_final>0.005)
    #print(sensitive[0])
    #diagonal_sensitive = []
    #for x,y in zip(sensitive[0], sensitive[1]):
    #    if x==y:
    #        diagonal_sensitive.append(x)
    #print(diagonal_sensitive)
    #with open(savepath+"sensitive_regions_0.005_alpha"+str(alpha)+".txt", "w") as output:
    #    output.write(str(diagonal_sensitive))
    else: 
   #plotting 
        plt.figure()
        plt.imshow(ak_final,vmax = 0.05,  vmin = 0) #vmax = 1,
        plt.colorbar()
        plt.savefig(savepath+'ak_final_2d_max0.05_'+str(alpha)+'.png')
    


#plot_averaging_kernel(Inversion, 1.1116017136941747e-06, 27, 6)


#for week in range(48,53):
 #   plot_prior_spatially(Inversion, week)
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
    #print(predictions)
    #total_spatial_result = xr.Dataset()
    plt.rcParams.update({'font.size':13})   
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
            plt.savefig(savepath+'Diff_to_prior_week_'+str(week)+'_alpha'+str(alpha)+'_xerr.png')
        else: 
            spatial_result = (spatial_result*12*10**(6))
            spatial_result.plot(x = 'longitude', y = 'latitude', cmap = 'seismic',vmin = vminv,cbar_kwargs = {'label' : r'flux [$\mu$gC m$^{-2}$ s$^{-1}$]'})
            plt.title('Week '+str(week))
            plt.savefig(savepath+'Spatial_results_week_'+str(week)+'_alpha'+str(alpha)+'_xerr.png')
            spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week)+'_'+str(alpha)+'.nc')
    #total_spatial_result.to_netcdf(path =savepath+'spatial_results_week_'+str(week_min)+'_'+str(week_max)+'.pkl')
    #print(predictions)
    #get_total(Inversion)



def do_everything(savepath, Inversion, mask_datapath_and_name, week_min, week_max, week_num):
    class_num = plot_input_mask(savepath,mask_datapath_and_name)
    #l1, l2 = find_two_optimal_lambdas(Inversion,[3e-5,1], 1e-9)# 1e-8
    l1 =  1.271463009778368e-05
    l3 = 0.0009414995908741231
    l2 = 0.0006228666734243309
    for l in [l1, l2, l3]: 
        #plot_l_curve(Inversion,savepath, l)
        plot_prior_spatially(Inversion,week_min, week_max, savepath)
        print('Plotting spatial results')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,l,vminv=None, diff =False)
        print('Plotting spatial difference of results to prior')
        plot_spatial_flux_results_or_diff_to_prior(savepath, Inversion, week_min, week_max,l,vminv=None, diff =True)
        print('Plotting averaging kernels')
        plot_averaging_kernel(Inversion, l, class_num, week_num,savepath)

import time
def isSorted(inList):
    length = len(inList)
    for i in range(length-1):
        if inList[i][1] != '' and inList[i+1][1] != '':
            date1 = time.strptime(inList[i][1], '%m-%d %H:%M:%S')
            date2 = time.strptime(inList[i+1][1], '%m-%d %H:%M:%S')
            if date1 > date2:
                return False
    return True

def sortList(inList):
    length = len(inList)
    for i in range(length):
        if inList[i][1] == '':
            continue
        for j in range(length-1,0,-1):
            if inList[j][1] == '':
                continue
            if i != j:
                date1 = time.strptime(inList[i][1], '%m-%d %H:%M:%S')
                date2 = time.strptime(inList[j][1], '%m-%d %H:%M:%S')
                if date2 < date1:
                    currentElem = inList[j]
                    inList.remove(inList[j]) 
                    inList.insert(i,currentElem)
        if isSorted(inList):
            break
    return inList

##### Versuch Gesamtemissionen nach inversion zu berechnen 
def calc_concentrations(Inversion, alpha, week, savepath):
    
    err = Inversion.get_land_ocean_error(1/10)
    predictions = Inversion.fit(alpha = alpha, xerr = err) 
    concentration_results = Inversion.footprints_flat.values*Inversion.predictions_flat.values*10**6
    conc_sum = concentration_results.sum(axis = 1)

    datapath_predictions = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/'
    ds = pd.read_pickle(datapath_predictions+'predictions.pkl')
    print(ds.keys())
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
    ax.errorbar(x= datetime(year =2020, month = 1, day = 7), y = 407.5, yerr = ds['measurement_uncertainty'].mean(), marker = '.',markersize = 7,linestyle='None',color = 'dimgrey')
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
    plt.savefig(savepath+'concentrations_results_err_on_bottom_measnum_equal_spacing.png')

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


savepath = '/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/Images/Setup_gridded/'

Inversion = InversionBioclass(
    result_path="/work/bb1170/RUN/b382105/Flexpart/TCCON/output/one_hour_runs/CO2/splitted/predictions.pkl",
    month="2019-12", 
    flux_path="/work/bb1170/RUN/b382105/Data/CarbonTracker2022/Flux/CT2022.flux1x1.",
    bioclass_path= "/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
    time_coarse = None,
    boundary=[110.0, 155.0, -45.0, -10.0],
    data_outside_month=False
)


plot_l_curve(Inversion, savepath, 6.2e-4)
#l1,l2 = find_two_optimal_lambdas(Inversion,[1e-8,1], 1e-9)
#calc_concentrations(Inversion,l2,49, savepath)
#plot_averaging_kernel(Inversion,  0.0006228666734243309, 699, 6,savepath, True)
#calc_concentrations(Inversion,  0.0006228666734243309,52, savepath)



#plot_input_mask(savepath,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1", selection= regions_sensitive)
#do_everything(savepath, Inversion,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
#              1,1,6)
#do_everything(savepath, Inversion,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version8_all1x1",
#              48,52,6)
#class_num = plot_input_mask(savepath,"/home/b/b382105/ColumnFLEXPART/resources/OekomaskAU_Flexpart_version7_Tasmania")#,selection = [5,6,7,8,16,20,31])
# for gridded : Optimal lambda 1: 1.271463009778368e-05
#Optimal lambda 2: 1.2802671525246307e-05
#Optimal lambda 3: 0.0006228666734243309
#Optimal lambda 4: 0.0009414995908741231